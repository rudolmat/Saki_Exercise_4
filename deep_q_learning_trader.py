import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
import time
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 2
        self.action_size = 9
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action_a = None
        self.last_action_b = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        # neuronales netz
        self.model = None

        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def calc_reward(self, current_portfolio_value: int):

        if current_portfolio_value > self.last_portfolio_value:
            return 1
        elif current_portfolio_value < self.last_portfolio_value:
            return -1
        else:
            return 0

    def action_mapping(self, recommendation):

        # Transform action into number
        action_dict = {"SELLSELL": 0, "SELLBUY": 1, "BUYSELL": 2, "BUYBUY": 3, "SELLHOLD": 4, "BUYHOLD": 5, "HOLDSELL": 6, "HOLDBUY": 7, "HOLDHOLD": 8}
        return action_dict[recommendation[0] + recommendation[1]]

    def comb_cash_share(self, temp_csc, stock_market_data):

        # Transform combination of cash, share_company_a, company_b into number; t=True, f=False
        comb_dict = {'ttt': 0, 'ttf': 1, 'tft': 2, 'tff': 3, 'ftt': 4, 'ftf': 5, 'fft': 6, 'fff': 7}   
        if temp_csc[0] >= max([stock_market_data.get_most_recent_price(Company.A), stock_market_data.get_most_recent_price(Company.B)]):
            tmp = "t"
        else:
            tmp = "f"
        if temp_csc[1] > 0:
            tmp += "t"
        else:
            tmp += "f"
        if temp_csc[2] > 0:
            tmp += "t"
        else:
            tmp += "f"
        return comb_dict[tmp]

    def states_compution(self, stock_market_data: StockMarketData, portfolio: Portfolio):

        temp_opinion = [self.expert_a.vote(stock_market_data[Company.A]), self.expert_b.vote(stock_market_data[Company.B])]
        for i, entry in enumerate(temp_opinion):
            if entry == Vote.BUY:
                temp_opinion[i] = "BUY"
            elif entry == Vote.SELL:
                temp_opinion[i] = "SELL"
            elif entry == Vote.HOLD:
                temp_opinion[i] = "HOLD"
        recommended_action = self.action_mapping(temp_opinion)

        cash = portfolio.cash
        share_a = portfolio.get_stock(Company.A)
        share_b = portfolio.get_stock(Company.A)
        cash_share_combination = self.comb_cash_share([cash, share_a, share_b], stock_market_data)

        state = np.array([recommended_action, cash_share_combination])
        state = np.reshape(state, [1, 2])
        return state

    def train_model(self):

        selected_batch = random.sample(self.memory, self.batch_size)
        target_list = np.zeros(self.batch_size, dtype=object)
        reward_list = np.zeros(self.batch_size, dtype=object)
        list_lastState = []
        list_currentState = []

        # save necessary information into lists
        index = 0
        for lastState, lastAction, currentReward, currentState in selected_batch:
            list_lastState.append(lastState[0])
            list_currentState.append(currentState[0])
            reward_list[index] = currentReward
            index += 1

        # predict rewards for old and current States
        y = self.model.predict(np.array(list_lastState))
        x = self.model.predict(np.array(list_currentState))

        for i in range(0, self.batch_size):
            target_list[i] = reward_list[i] + self.gamma * np.amax(x[i])
            y[i][selected_batch[i][1]] = target_list[i]

        # decrease epsilon until min is reached for trade-off between exploration and exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

        self.model.train_on_batch(np.array(list_lastState), y)

    def append_memory(self, last_state, last_action: int, reward: int, current_state):

        self.memory.append((last_state, last_action, reward, current_state))

    def select_action(self, current_state, current_portfolio_value):

        # Exploration
        if random.randint(0, 1) <= self.epsilon:
            current_action = random.randint(0, self.action_size-1)

        # Exploitation
        else:
            current_action = np.argmax(self.model.predict(current_state))

        return current_action

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation


        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        current_state = self.states_compution(stock_market_data, portfolio)
        current_portfolio_value = portfolio.get_value(stock_market_data)

        if self.train_while_trading is False:
            # for test set use trained ann
            action = np.argmax(self.model.predict(current_state)[0])

        else:
            action = self.select_action(current_state, current_portfolio_value)
            if self.last_state is not None:
                reward = self.calc_reward(current_portfolio_value)
                self.append_memory(self.last_state, self.last_action_a, reward, current_state)
                # train model as soon as sufficient memory is reached
                if len(self.memory) > self.min_size_of_memory_before_training:
                    self.train_model()

        # Split action into individual actions for Company A and B
        current_action_a = 0
        current_action_b = 0
        assert action < 9 and action >= 0
        if action == 0:
            current_action_a = OrderType.SELL
            current_action_b = OrderType.SELL
            amount_to_sell_a = portfolio.get_stock(Company.A)
            amount_to_sell_b = portfolio.get_stock(Company.B)
        elif action == 1:
            current_action_a = OrderType.SELL
            amount_to_sell_a = portfolio.get_stock(Company.A)
            current_action_b = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_b = int(portfolio.cash/stock_price)
        elif action == 2:
            current_action_a = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int(portfolio.cash/stock_price)
            current_action_b = OrderType.SELL
            amount_to_sell_b = portfolio.get_stock(Company.B)
        elif action == 3:
            current_action_a = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int((portfolio.cash/stock_price)/2)
            current_action_b = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.B)
            amount_to_buy_b = int((portfolio.cash/stock_price)/2)
        elif action == 4:
            current_action_a = OrderType.SELL
            amount_to_sell_a = portfolio.get_stock(Company.A)
            # current_action_b = "hold"
        elif action == 5:
            current_action_a = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.A)
            amount_to_buy_a = int(portfolio.cash/stock_price)
            # current_action_b = "hold"
        elif action == 6:
            # current_action_a = "hold"
            current_action_b = OrderType.SELL
            amount_to_sell_b = portfolio.get_stock(Company.B)
        elif action == 7:
            # current_action_a = "hold"
            current_action_b = OrderType.BUY
            stock_price = stock_market_data.get_most_recent_price(Company.B)
            amount_to_buy_b = int(portfolio.cash/stock_price)

        order_list = []

        if current_action_a != 0:
            if current_action_a == OrderType.SELL and amount_to_sell_a > 0:
                order_list.append(Order(current_action_a, Company.A, amount_to_sell_a))
            elif current_action_a == OrderType.BUY and portfolio.cash > 0:
                order_list.append(Order(current_action_a, Company.A, amount_to_buy_a))

        if current_action_b != 0:
            if current_action_b == OrderType.SELL and amount_to_sell_b > 0:
                order_list.append(Order(current_action_b, Company.B, amount_to_sell_b))
            elif current_action_b == OrderType.BUY and portfolio.cash > 0:
                order_list.append(Order(current_action_b, Company.B, amount_to_buy_b))

        self.last_action_a = action
        self.last_state = current_state
        self.last_portfolio_value = current_portfolio_value
        return order_list

# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    beginning = time.time()
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []
    print("Episode over")
    print(final_values_training)
    print(final_values_test)

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")
    end = time.time()
    print(end-beginning)
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()

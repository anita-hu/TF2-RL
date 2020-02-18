import random
import numpy as np
import pickle
import gym
import imageio  # write env render to mp4
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with LSTM layer followed by Dense layers
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env
'''


class DQN:
    def __init__(
            self, 
            env, 
            memory_cap=1000,
            time_steps=4,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=32,
            tau=0.125
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.rewards = [0]
        self.q_values = []

    def create_model(self):
        model = Sequential()
        model.add(LSTM(24, input_shape=(self.time_steps, self.state_shape[0]), activation="tanh"))
        model.add(Dense(16))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def act(self, test=False):
        states = np.expand_dims(self.stored_states, axis=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.q_values.append(max(q_values))
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # sample random action
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        batch_states = []
        batch_target = []
        for sample in samples:  # calculate target y for each sample
            states, action, reward, new_states, done = sample
            batch_states.append(states)
            states = np.expand_dims(states, axis=0)
            target = self.target_model.predict(states)[0]
            if done:
                target[action] = reward
            else:
                new_states = np.expand_dims(new_states, axis=0)
                q_future = max(self.target_model.predict(new_states)[0])
                target[action] = reward + q_future * self.gamma
            batch_target.append(target)
        self.model.fit(np.array(batch_states), np.array(batch_target), epochs=1, verbose=0)

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.model.save(fn)

    def load_model(self, fn):
        # load model from .h5 file
        self.model = tf.keras.models.load_model(fn)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=10, max_steps=500, save_freq=10):
        done, episode, steps = True, 0, 0
        while episode < max_episodes:
            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                self.save_model("dqn_lstm_maxed_episode{}_time_step{}.h5".format(episode, self.time_steps))

            if done:
                self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))  # clear stored states
                done, cur_state, steps = False, self.env.reset(), 0
                self.update_states(cur_state)  # update stored states
                self.rewards.append(0)
                print("episode {}: {} reward".format(episode, self.rewards[-2]))
                if episode % save_freq == 0:  # save model every n episodes
                    self.save_model("dqn_lstm_episode{}_time_step{}.h5".format(episode, self.time_steps))
                episode += 1

            action = self.act()  # model determine action, states taken from self.stored_states
            new_state, reward, done, _ = self.env.step(action)  # perform action on env
            modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
            prev_stored_states = self.stored_states
            self.update_states(new_state)  # update stored states
            self.remember(prev_stored_states, action, modified_reward, self.stored_states, done)
            self.replay()  # iterates default (prediction) model through memory replay
            self.target_update()  # iterates target model

            self.rewards[-1] += reward  # sum reward
            steps += 1

        self.save_model("dqn_lstm_final_episode{}_time_step{}.h5".format(episode, self.time_steps))

    def test(self, render=True, fps=30, filename='test_render_330.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(test=True)
            new_state, reward, done, _ = self.env.step(action)
            self.update_states(new_state)
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards

    def plot_q_values(self):
        file = open('q_values_{}_states.txt'.format(self.time_steps), 'wb')
        pickle.dump(self.q_values, file)
        file.close()
        
        plt.clf()
        plt.plot(self.q_values, label="Q value")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('q_values_{}_states.png'.format(self.time_steps), bbox_inches='tight')

    def plot_rewards(self):
        file = open('rewards_{}_states.txt'.format(self.time_steps), 'wb')
        pickle.dump(self.rewards[:-1], file)
        file.close()
        
        plt.clf()
        plt.plot(self.rewards, label="Reward")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('rewards_{}_states.png'.format(self.time_steps), bbox_inches='tight')


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 500
    dqn_agent = DQN(env, time_steps=4)
    # dqn_agent.load_model("lstm_models/time_step4/dqn_lstm_episode50_time_step4.h5")
    # rewards = dqn_agent.test()
    # print("Total rewards: ", rewards)
    dqn_agent.train(max_episodes=50)

    dqn_agent.plot_q_values()
    dqn_agent.plot_rewards()

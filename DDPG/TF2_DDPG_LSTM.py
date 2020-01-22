# Paper: https://arxiv.org/pdf/1509.02971.pdf

import gym
import random
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

# TODO: improvements
# - normalize observations
# - add support in agent for discrete-continuous action space


tf.keras.backend.set_floatx('float64')


def actor(input_shape, action_dim, action_bound, units=(24, 16), num_actions=1):
    states = Input(shape=input_shape)
    state_features = LSTM(units[0], name="state_lstm", activation="tanh")(states)
    x = Dense(units[1], name="fc1", activation='relu')(state_features)
    # x = Dense(units[2], name="fc2", activation='relu')(x)
    outputs = []  # for loop for discrete-continuous action space
    for i in range(num_actions):
        unscaled_output = Dense(action_dim, name="output{}".format(i), activation='relu')(x)
        scalar = action_bound * np.ones(action_dim)
        outputs.append(Lambda(lambda op: op * scalar)(unscaled_output))

    model = Model(inputs=states, outputs=outputs)

    return model


def critic(input_shape, action_dim, units=(24, 16), num_actions=1):
    inputs = [Input(shape=input_shape)]
    for i in range(num_actions):  # for loop for discrete-continuous action space
        inputs.append(Input(shape=(action_dim,)))
    state_features = LSTM(units[0], name="state_lstm", activation="tanh")(inputs[0])
    concat = Concatenate(axis=-1)([state_features]+inputs[1:])
    x = Dense(units[1], name="fc1", activation='relu')(concat)
    # x = Dense(units[2], name="fc2", activation='relu')(x)
    output = Dense(1, name="output")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


class DDPG:
    def __init__(
            self,
            env,
            discrete=False,
            time_steps=4,
            lr_actor=1e-5,
            lr_critic=1e-3,
            actor_units=(24, 16),
            critic_units=(24, 16),
            sigma=0.4,
            sigma_decay=5e-4,
            tau=0.125,
            gamma=0.85,
            batch_size=64,
            memory_cap=1000
    ):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.action_dim = env.action_space.n
        input_shape = (self.time_steps, self.state_shape[0])
        self.stored_states = np.zeros(input_shape)
        self.discrete = discrete
        self.action_bound = env.action_space.high if not discrete else 1.
        self.memory = deque(maxlen=memory_cap)

        # Define and initialize Actor network
        self.actor = actor(input_shape, self.action_dim, self.action_bound, actor_units)
        self.actor_target = actor(input_shape, self.action_dim, self.action_bound, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.)

        # Define and initialize Critic network
        self.critic = critic(input_shape, self.action_dim, critic_units)
        self.critic_target = critic(input_shape, self.action_dim, critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)
        update_target_weights(self.critic, self.critic_target, tau=1.)

        # Set hyperparameters
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Training log
        self.rewards = [0]
        self.q_values = []

    def update_states(self, new_state):
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def act(self, add_noise=True):
        states = np.expand_dims(self.stored_states, axis=0).astype(np.float32)
        a = self.actor.predict(states)
        # add mean-zero Gaussian noise
        a += tf.random.normal(shape=a.shape, mean=0., stddev=self.sigma * float(add_noise),
                              dtype=tf.float32)*self.action_bound
        a = tf.clip_by_value(a, -self.action_bound, self.action_bound)
        q_val = self.critic.predict([states, a])
        self.q_values.append(q_val[0][0])
        self.sigma *= (1 - self.sigma_decay)

        return a

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor = load_model(a_fn)
        self.actor_target = load_model(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic = load_model(c_fn)
        self.critic_target = load_model(c_fn)
        print(self.critic.summary())

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        batch_y = []
        batch_states = []
        batch_target_actions = []
        for sample in samples:
            state, action, reward, next_state, done = sample
            next_action = self.actor_target.predict(next_state)
            q_future = self.critic_target.predict([next_state, next_action])[0]
            y = reward + q_future * self.gamma * (1. - done)
            batch_states.append(state[0])
            batch_y.append(y)
            batch_target_actions.append(action[0])

        # train critic
        batch_states = np.array(batch_states)
        self.critic.fit([batch_states, np.array(batch_target_actions)], np.array(batch_y), epochs=1,
                        batch_size=self.batch_size, verbose=0)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(batch_states)
            actor_loss = -tf.reduce_mean(self.critic([batch_states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def train(self, max_episodes=50, max_epochs=8000, max_steps=500, resume_episode=0, save_freq=50):
        done, episode, cur_state, steps, epoch = False, resume_episode, self.env.reset(), 0, 0
        while episode < max_episodes or epoch < max_epochs:
            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                "ddpg_critic_episode{}.h5".format(episode))

            if done:
                episode += 1
                print("episode {}: {} reward, {} q value, {} sigma, {} epochs".format(
                    episode, self.rewards[-1], self.q_values[-steps], self.sigma, epoch))
                done, cur_state, steps = False, self.env.reset(), 0
                self.rewards.append(0)
                if episode % save_freq == 0:
                    self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                    "ddpg_critic_episode{}.h5".format(episode))

            a = self.act()
            action = np.argmax(a) if self.discrete else a[0]
            next_state, reward, done, _ = self.env.step(action)
            modified_reward = 1 - abs(next_state[2] / (np.pi / 2))  # divide by 90 degrees
            cur_stored_states = self.stored_states
            self.update_states(next_state)

            self.remember(cur_stored_states, a, modified_reward, self.stored_states, done)
            self.replay()

            update_target_weights(self.actor, self.actor_target, tau=self.tau)
            update_target_weights(self.critic, self.critic_target, tau=self.tau)

            self.rewards[-1] += reward
            steps += 1
            epoch += 1

        self.save_model("ddpg_actor_final_episode{}.h5".format(episode),
                        "ddpg_critic_final_episode{}.h5".format(episode))

    def plot_rewards(self):
        file = open('rewards.txt', 'wb')
        pickle.dump(self.rewards, file)
        file.close()

        plt.clf()
        plt.plot(self.rewards[:-1], label="Reward")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('rewards.png', bbox_inches='tight')

    def plot_q_values(self):
        file = open('q_values.txt', 'wb')
        pickle.dump(self.q_values, file)
        file.close()

        plt.clf()
        plt.plot(self.q_values, label="Q value")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('q_values.png', bbox_inches='tight')


if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1")
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = DDPG(gym_env, discrete=is_discrete)
    # ddpg.load_critic("model/ddpg_critic_episode200.h5")
    ddpg.train(max_episodes=50)
    ddpg.plot_rewards()
    ddpg.plot_q_values()

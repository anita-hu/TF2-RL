import gym
import random
import imageio
import pickle
import numpy as np
from collections import deque
import threading
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

'''
Original paper: https://arxiv.org/pdf/1903.00827.pdf
'''

tf.keras.backend.set_floatx('float64')


def actor(state_shape, action_dim, action_bound, units=(400, 300), num_actions=1):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L1", activation='relu')(state)
    x = Dense(units[1], name="L2", activation='relu')(x)
    # x = Dense(units[2], name="L3", activation='relu')(x)
    outputs = []  # for loop for discrete-continuous action space
    for i in range(num_actions):
        unscaled_output = Dense(action_dim, name="L{}".format(i), activation='tanh')(x)
        scalar = action_bound * np.ones(action_dim)
        output = Lambda(lambda op: op * scalar)(unscaled_output)
        # output = Lambda(lambda op: op + shift)(output)  # for action range not centered at zero
        outputs.append(output)

    model = Model(inputs=state, outputs=outputs)

    return model


def critic(state_shape, action_dim, units=(48, 24), num_actions=1):
    inputs = [Input(shape=state_shape)]
    for i in range(num_actions):  # for loop for discrete-continuous action space
        inputs.append(Input(shape=(action_dim,)))
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L1", activation='relu')(concat)
    x = Dense(units[1], name="L2", activation='relu')(x)
    # x = Dense(units[2], name="L3", activation='relu')(x)
    output = Dense(1, name="L4")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


def is_threads_alive(threads):
    for t in threads:
        if t.isAlive():
            return 1
    return 0


class AE_DDPG:
    def __init__(
            self,
            env_name,
            num_envs=5,
            discrete=False,
            lr_actor=1e-4,
            lr_critic=1e-3,
            actor_units=(24, 16),
            critic_units=(24, 16),
            sigma=0.4,
            sigma_decay=0.9995,
            tau=0.125,
            gamma=0.85,
            rho=0.2,
            batch_size=64,
            memory_cap=1500,
            hmemory_cap=1000,
            cache_size=500
    ):
        self.env = [gym.make(env_name) for i in range(num_envs)]
        self.num_envs = num_envs  # number of environments for async data collection
        self.state_shape = self.env[0].observation_space.shape  # shape of observations
        self.action_dim = self.env[0].action_space.n  # number of actions
        self.discrete = discrete
        # action bound assumes range is centered at zero (e.g. -10 to 10)
        # add another layer at the end of the actor model to shift action values if it is not centered at zero
        self.action_bound = self.env[0].action_space.high if not discrete else 1.

        # Initialize memory buffers
        self.memory = deque(maxlen=memory_cap)
        self.hmemory = deque(maxlen=hmemory_cap)
        self.cache_buffer = [deque(maxlen=cache_size) for i in range(num_envs)]

        # Define and initialize Actor network
        self.actor = actor(self.state_shape, self.action_dim, self.action_bound, actor_units)
        self.actor_target = actor(self.state_shape, self.action_dim, self.action_bound, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        update_target_weights(self.actor, self.actor_target, tau=1.)

        # Define and initialize Critic network
        self.critic = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_target = critic(self.state_shape, self.action_dim, critic_units)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)
        update_target_weights(self.critic, self.critic_target, tau=1.)

        # Set hyperparameters
        self.sigma = sigma  # stddev for mean-zero gaussian noise
        # expotential decay, we take into account it is updated by multiple threads
        self.sigma_decay = sigma_decay**(1/self.num_envs)
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.rho = rho  # chance of sampling from hmemory, paper recommends [0.05, 0.25]
        self.batch_size = batch_size

        # Training log
        self.max_reward = 0
        self.q_values = [[]] * self.num_envs
        self.rewards = [[]] * self.num_envs

    def act(self, state, random_walk):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        a = self.actor.predict(state)
        # add random walk noise
        a += random_walk * self.action_bound
        a = tf.clip_by_value(a, -self.action_bound, self.action_bound)

        q_val = self.critic.predict([state, a])
        self.sigma *= self.sigma_decay

        return a, q_val[0][0]

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
        print(self.critic.summary())

    def store_memory(self, state, action, reward, next_state, done, hmemory=False):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        if hmemory:
            self.hmemory.append([state, action, reward, next_state, done])
        else:
            self.memory.append([state, action, reward, next_state, done])

    def sample_batch(self):
        samples = []
        for i in range(self.batch_size):
            if np.random.random() < self.rho and len(self.hmemory) > 0:
                sample = random.sample(self.hmemory, 1)  # sample from highly rewarded trajectories
            else:
                sample = random.sample(self.memory, 1)
            samples.append(sample[0])

        return samples

    def replay(self):
        samples = self.sample_batch()
        batch_y = []
        batch_states = []
        batch_target_actions = []
        for sample in samples:  # calculate target y for each sample
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

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def async_collection(self, index):
        while True:
            done, cur_state, total_reward, rand_walk_noise = False, self.env[index].reset(), 0, 0
            while not done:
                a, q_val = self.act(cur_state, rand_walk_noise)  # model determine action given state
                action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
                next_state, reward, done, _ = self.env[index].step(action)  # perform action on env
                modified_reward = 1 - abs(
                    next_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle

                self.cache_buffer[index].append([cur_state, a, modified_reward, next_state, done])  # add to buffer
                self.store_memory(cur_state, a, modified_reward, next_state, done)  # add to memory

                cur_state = next_state
                total_reward += reward
                rand_walk_noise += tf.random.normal(shape=a.shape, mean=0., stddev=self.sigma, dtype=tf.float32)
                self.q_values[index].append(q_val)

            self.rewards[index].append(total_reward)

            if total_reward > self.max_reward:
                self.max_reward = total_reward
                while len(self.cache_buffer[index]) > 0:
                    transitions = self.cache_buffer[index].pop()
                    self.store_memory(*transitions, hmemory=True)

    def train(self, max_epochs=8000, save_freq=50):
        collection_threads, epoch = [], 0
        self.actor._make_predict_function()  # from https://github.com/jaromiru/AI-blog/issues/2
        print("-- starting threads --")
        for i in range(self.num_envs):
            t = threading.Thread(target=self.async_collection, args=(i, ), daemon=True)
            t.start()
            collection_threads.append(t)

        print("-- waiting for first batch of data --")
        while is_threads_alive(collection_threads) and epoch < max_epochs:
            if len(self.memory) < self.batch_size:
                continue

            self.replay()  # train models through memory replay
            update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
            update_target_weights(self.critic, self.critic_target, tau=self.tau)

            epoch += 1
            print("train epoch {}: {} max reward, {} q value, {} sigma".format(epoch, self.max_reward,
                                                                               self.q_values[0][-1], self.sigma))

            if epoch % save_freq == 0:
                self.save_model("ddpg_actor_epoch{}.h5".format(epoch),
                                "ddpg_critic_epoch{}.h5".format(epoch))

        print("-- saving final model --")
        self.save_model("ddpg_actor_final_epoch{}.h5".format(max_epochs),
                        "ddpg_critic_final_epoch{}.h5".format(max_epochs))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env[0].reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            a, _ = self.act(cur_state, 0)
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env[0].step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env[0].render(mode='rgb_array'))
        video.close()
        print("Total rewards: ", rewards)

    def plot_rewards(self):
        file = open('rewards.txt', 'wb')
        pickle.dump(self.rewards, file)
        file.close()

        plt.clf()
        plt.plot(self.rewards[0], label="Reward")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('rewards.png', bbox_inches='tight')

    def plot_q_values(self):
        file = open('q_values.txt', 'wb')
        pickle.dump(self.q_values, file)
        file.close()

        plt.clf()
        plt.plot(self.q_values[0], label="Q value")
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('q_values.png', bbox_inches='tight')


if __name__ == "__main__":
    name = "CartPole-v1"
    gym_env = gym.make(name)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = AE_DDPG(name, discrete=is_discrete)
    # ddpg.load_critic("ddpg_critic_epoch950.h5")
    # ddpg.load_actor("ddpg_actor_epoch950.h5")
    ddpg.train(max_epochs=5000)
    # ddpg.test()
    ddpg.plot_rewards()
    ddpg.plot_q_values()

import gym
import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

# Original paper: https://arxiv.org/pdf/1509.02971.pdf

tf.keras.backend.set_floatx('float64')


def actor(state_shape, action_dim, action_bound, action_shift, units=(400, 300), num_actions=1):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation='relu')(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)

    outputs = []  # for loop for discrete-continuous action space
    for i in range(num_actions):
        unscaled_output = Dense(action_dim, name="Out{}".format(i), activation='tanh')(x)
        scalar = action_bound * np.ones(action_dim)
        output = Lambda(lambda op: op * scalar)(unscaled_output)
        if np.sum(action_shift) != 0:
            output = Lambda(lambda op: op + action_shift)(output)  # for action range not centered at zero
        outputs.append(output)

    model = Model(inputs=state, outputs=outputs)

    return model


def critic(state_shape, action_dim, units=(48, 24), num_actions=1):
    inputs = [Input(shape=state_shape)]
    for i in range(num_actions):  # for loop for discrete-continuous action space
        inputs.append(Input(shape=(action_dim,)))
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L0", activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
    output = Dense(1, name="Out")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


class DDPG:
    def __init__(
            self,
            env,
            discrete=False,
            lr_actor=1e-5,
            lr_critic=1e-3,
            actor_units=(24, 16),
            critic_units=(24, 16),
            sigma=0.4,
            sigma_decay=0.9995,
            tau=0.125,
            gamma=0.85,
            batch_size=64,
            memory_cap=1000
    ):
        self.env = env
        self.state_shape = env.observation_space.shape  # shape of observations
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]  # number of actions
        self.discrete = discrete
        self.action_bound = (env.action_space.high - env.action_space.low) / 2 if not discrete else 1.
        self.action_shift = (env.action_space.high + env.action_space.low) / 2 if not discrete else 0.
        self.memory = deque(maxlen=memory_cap)

        # Define and initialize Actor network
        self.actor = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
        self.actor_target = actor(self.state_shape, self.action_dim, self.action_bound, self.action_shift, actor_units)
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
        self.sigma_decay = sigma_decay  # expotential decay
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # Tensorboard
        self.summaries = {}

    def act(self, state, add_noise=True):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        a = self.actor.predict(state)
        # add mean-zero Gaussian noise
        a += tf.random.normal(shape=a.shape, mean=0., stddev=self.sigma * float(add_noise),
                              dtype=tf.float32) * self.action_bound
        a = tf.clip_by_value(a, -self.action_bound + self.action_shift, self.action_bound + self.action_shift)

        self.summaries['q_val'] = self.critic.predict([state, a])[0][0]
        self.sigma *= self.sigma_decay

        return a

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

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]
        next_actions = self.actor_target.predict(next_states)
        q_future = self.critic_target.predict([next_states, next_actions])
        target_qs = rewards + q_future * self.gamma * (1. - dones)

        # train critic
        hist = self.critic.fit([states, actions], target_qs, epochs=1, batch_size=self.batch_size, verbose=0)

        # train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # tensorboard info
        self.summaries['critic_loss'] = np.mean(hist.history['loss'])
        self.summaries['actor_loss'] = actor_loss

    def train(self, max_episodes=50, max_epochs=8000, max_steps=1000, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DDPG_basic_' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        cur_state = self.env.reset()
        while episode < max_episodes or epoch < max_epochs:
            if steps > max_steps:
                print("episode {}, reached max steps".format(episode))
                self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                "ddpg_critic_episode{}.h5".format(episode))

            if done:
                episode += 1
                print("episode {}: {} total reward, {} sigma, {} steps, {} epochs".format(
                    episode, total_reward, self.sigma, steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/total_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/steps', steps, step=episode)

                summary_writer.flush()

                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                if episode % save_freq == 0:
                    self.save_model("ddpg_actor_episode{}.h5".format(episode),
                                    "ddpg_critic_episode{}.h5".format(episode))

            a = self.act(cur_state)  # model determine action given state
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env.step(action)  # perform action on env
            modified_reward = 1 - abs(next_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle

            self.remember(cur_state, a, modified_reward, next_state, done)  # add to memory
            self.replay()  # train models through memory replay

            update_target_weights(self.actor, self.actor_target, tau=self.tau)  # iterates target model
            update_target_weights(self.critic, self.critic_target, tau=self.tau)

            cur_state = next_state
            total_reward += reward
            steps += 1
            epoch += 1

            # Tensorboard update
            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/critic_loss', self.summaries['critic_loss'], step=epoch)
                tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
                tf.summary.scalar('Stats/sigma', self.sigma, step=epoch)

            summary_writer.flush()

        self.save_model("ddpg_actor_final_episode{}.h5".format(episode),
                        "ddpg_critic_final_episode{}.h5".format(episode))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            a = self.act(cur_state, add_noise=False)
            action = np.argmax(a) if self.discrete else a[0]  # post process for discrete action space
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


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
    # ddpg.load_critic("basic_models/ddpg_critic_final_episode104.h5")
    # ddpg.load_actor("basic_models/ddpg_actor_final_episode104.h5")
    ddpg.train(max_episodes=1000)
    # rewards = ddpg.test()
    # print("Total rewards: ", rewards)

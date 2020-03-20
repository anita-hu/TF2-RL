import gym
import copy
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.keras.backend.set_floatx('float64')

# paper https://arxiv.org/pdf/1707.06347.pdf
# code references https://github.com/uidilr/ppo_tf/blob/master/ppo.py,
# https://github.com/openai/baselines/tree/master/baselines/ppo1


def model(state_shape, action_dim, units=(400, 300, 100), discrete=False):
    state = Input(shape=state_shape)

    vf = Dense(units[0], name="Value_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        vf = Dense(units[index], name="Value_L{}".format(index), activation="tanh")(vf)

    value_pred = Dense(1, name="Out_value")(vf)

    pi = Dense(units[0], name="Policy_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        pi = Dense(units[index], name="Policy_L{}".format(index), activation="tanh")(pi)

    if discrete:
        action_probs = Dense(action_dim, name="Out_probs", activation='softmax')(pi)
        model = Model(inputs=state, outputs=[action_probs, value_pred])
    else:
        actions_mean = Dense(action_dim, name="Out_mean", activation='tanh')(pi)
        model = Model(inputs=state, outputs=[actions_mean, value_pred])

    return model


class PPO:
    def __init__(
            self,
            env,
            discrete=False,
            lr=3e-4,
            hidden_units=(24, 16),
            c1=1.0,
            c2=0.0,
            clip_ratio=0.2,
            gamma=0.99,
            batch_size=64,
            n_updates=4,
            memory_cap=100000
    ):
        self.env = env
        self.state_shape = env.observation_space.shape  # shape of observations
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]  # number of actions
        self.discrete = discrete
        if not discrete:
            self.action_bound = (env.action_space.high - env.action_space.low) / 2
            self.action_shift = (env.action_space.high + env.action_space.low) / 2
        self.memory = deque(maxlen=int(memory_cap))

        # Define and initialize network
        self.policy = model(self.state_shape, self.action_dim, hidden_units, discrete=discrete)
        self.model_optimizer = Adam(learning_rate=lr)
        print(self.policy.summary())

        # Stdev for continuous action
        if not discrete:
            self.policy_stdev = tf.Variable(tf.zeros(self.action_dim, dtype=tf.float64), trainable=True)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.n_updates = n_updates  # number of epochs per episode

        # Tensorboard
        self.summaries = {}

    def get_dist(self, output):
        if self.discrete:
            dist = tfd.Categorical(output)
        else:
            std = tf.math.exp(self.policy_stdev)
            dist = tfd.Normal(loc=output, scale=std)

        return dist

    def evaluate_actions(self, state, action):
        output, value = self.policy(state)
        dist = self.get_dist(output)

        log_probs = tf.reduce_sum(dist.log_prob(action))
        entropy = dist.entropy()

        return log_probs, entropy, value

    def act(self, state, test=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)
        output, value = self.policy.predict(state)
        dist = self.get_dist(output)

        if self.discrete:
            action = tf.math.argmax(output, axis=-1) if test else dist.sample()
        else:
            action = output if test else dist.sample()

        log_probs = tf.reduce_sum(dist.log_prob(action))

        if not self.discrete:
            action = action * self.action_bound + self.action_shift

        return action[0].numpy(), value[0][0], log_probs.numpy()

    def save_model(self, fn):
        self.policy.save(fn)

    def load_model(self, fn):
        self.policy.load_weights(fn)
        print(self.policy.summary())

    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def learn(self, observations, actions, log_probs, rewards, gaes):
        rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
        with tf.GradientTape(persistent=True) as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(tf.multiply(gaes, ratios), tf.multiply(gaes, clipped_ratios))
            loss_clip = -tf.reduce_mean(loss_clip)

            vf_loss = tf.reduce_mean(tf.math.square(state_values - rewards))

            entropy = -tf.reduce_mean(entropy)
            total_loss = loss_clip + self.c1 * vf_loss + self.c2 * entropy

        train_variables = self.policy.trainable_variables
        if not self.discrete:
            train_variables += [self.policy_stdev]
        grad = tape.gradient(total_loss, train_variables)  # compute actor gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, max_epochs=8000, max_steps=1000, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch = False, 0, 0, 0
        cur_state = self.env.reset()

        obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

        while epoch < max_epochs:
            while not done and steps < max_steps:
                action, value, log_prob = self.act(cur_state)  # determine action
                cur_state, reward, done, _ = self.env.step(action)  # act on env
                self.env.render(mode='rgb_array')

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            data = [obs, actions, log_probs, rewards, gaes]

            for i in range(self.n_updates):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(rewards), size=self.batch_size)
                sampled_data = [np.take(a=a, indices=sample_indices, axis=0) for a in data]

                # Train model
                self.learn(*sampled_data)

                # Tensorboard update
                with summary_writer.as_default():
                    tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=epoch)
                    tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
                    tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                    tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)

                summary_writer.flush()
                epoch += 1

            episode += 1
            print("episode {}: {} total reward, {} steps, {} epochs".format(
                episode, np.sum(rewards), steps, epoch))

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)

            summary_writer.flush()

            done, cur_state, steps, episode_reward = False, self.env.reset(), 0, 0
            obs, actions, rewards, v_preds, next_v_preds = [], [], [], [], []
            if episode % save_freq == 0:
                self.save_model("ppo_episode{}.h5".format(episode))

        self.save_model("ppo_final_episode{}.h5".format(episode))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(cur_state, test=True)
            next_state, reward, done, _ = self.env.step(action[0])
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    gym_env = gym.make("Pendulum-v0")
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ppo = PPO(gym_env, discrete=is_discrete)

    # ppo.load_model("ppo_episode480.h5")
    ppo.train(max_epochs=200000, save_freq=50)
    # reward = ppo.test()
    # print(reward)

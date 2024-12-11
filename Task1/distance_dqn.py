import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from gym.envs.toy_text.frozen_lake import generate_random_map
import argparse

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--ep', dest='ep', type=int, default=10000)
parser.add_argument('--decay', dest='decay', type=float, default=0.999993)
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9)
parser.add_argument('--lr', dest='lr', type=float, default=0.01)

args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

class DQNAgent():
    def __init__(self,
                  env=None,
                  discount_factor=0.90,
                  max_eps=1,
                  min_eps=0.1,
                  num_episodes=10000,
                  eps_decay=0.999993,
                  learning_rate=0.01
                  ):
        self.discount_factor = discount_factor
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.num_episodes = num_episodes
        self.rList = [] #Record reward
        self.alg_name = 'DQN'
        self.goal_reached_n = 0  # count the number of times the goal is reached
        self.eps = self.max_eps
        self.eps_decay = eps_decay
        self.q_table = np.zeros((100, 4))
        self.goal_position = np.array(np.where(env.desc == b'G')).flatten()
        self.position = np.array(np.where(env.desc == b'S')).flatten()
        self.max_distance = np.abs(self.position - self.goal_position).sum()
        self.env = env
        self.learning_rate = learning_rate

        self.qnetwork = self.get_model([None, 100])
        self.qnetwork.train()

        self.train_weights = self.qnetwork.trainable_weights
        self.optimizer = tf.optimizers.SGD(learning_rate=learning_rate)



    ##################### DQN ##########################
    def to_one_hot(self, i, n_classes=None):
        a = np.zeros(n_classes, 'uint8')
        a[i] = 1
        return a


    ## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
    # encoding for state: 10x10 grid can be represented by one-hot vector with 100 integers.
    def get_model(self, inputs_shape):
        ni = tl.layers.Input(inputs_shape, name='observation')
        nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
        return tl.models.Model(inputs=ni, outputs=nn)
    
    def save_ckpt(self, model):  # save trained weights
        path = os.path.join('model', '_distance_'.join([self.alg_name, self.learning_rate, self.discount_factor]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)

        with open(os.path.join(path, 'env_desc.txt'), 'w') as f:
            f.write(str(self.env.desc))


    def load_ckpt(self, model):  # load trained weights
        path = os.path.join('model', '_distance_'.join([self.alg_name, self.learning_rate, self.discount_factor]))
        tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)


    def train(self):
        flattened_map = self.env.desc.flatten()
        all_episode_reward = []
        for i in range(self.num_episodes):
            visited = set()
            ## Reset environment and get first new observation
            s = self.env.reset()[0]  # observation is state
            rAll = 0
            while True:
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = self.qnetwork(np.asarray([self.to_one_hot(s, 100)], dtype=np.float32)).numpy()
                self.q_table[s] = allQ
                a = np.argmax(allQ, 1)

                ## e-Greedy Exploration !!! sample random action
                if np.random.rand(1) < self.eps:
                    a[0] = self.env.action_space.sample()
                ## Get new state and reward from self.ironment
                next_state, r, d, _, _ = self.env.step(a[0])

                # if flattened_map[next_state] == b'G':
                #     self.goal_reached_n += 1
                self.position = np.array([next_state // 10, next_state % 10])
                
                manhattan_distance = np.abs(self.position - self.goal_position).sum()
                dist_reward = manhattan_distance / self.max_distance

                if next_state in visited:
                    r = -10
                elif next_state == s and not d:
                    r = -10
                elif flattened_map[next_state] == b'H':
                    r = -50
                elif flattened_map[next_state] == b'G':
                    r = 100
                    self.goal_reached_n += 1
                elif flattened_map[next_state] == b'F':
                    r = -1

                if r < 0:
                    r *= dist_reward
                
                ## Obtain the Q' values by feeding the new state through our network
                q1 = self.qnetwork(np.asarray([self.to_one_hot(next_state, 100)], dtype=np.float32)).numpy()
                ## Obtain maxQ' and set our target value for chosen action.
                max_q1 = np.max(q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                targetQ = allQ
                targetQ[0, a[0]] = r + self.discount_factor * max_q1
                ## Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                with tf.GradientTape() as tape:
                    _qvalues = self.qnetwork(np.asarray([self.to_one_hot(s, 100)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, self.train_weights)
                self.optimizer.apply_gradients(zip(grad, self.train_weights))

                rAll += r
                s = next_state
                
                if s not in visited:
                    visited.add(s)
                ## Reduce chance of random action if an episode is done.
                if d == True:
                    if self.eps > self.min_eps:
                        self.eps *= self.eps_decay
                    break

            ## Note that, the rewards here with random action
            if i % 10000 == 0:
                print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Epsilon: {:.4f} | Goal reached: {}' \
                        .format(i, self.num_episodes, rAll, self.eps, self.goal_reached_n))
            if i % 100000 == 0 :
                print(self.q_table)
            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

        self.save_ckpt(self.qnetwork)  # save model
        print(self.q_table)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([self.alg_name, self.learning_rate, self.discount_factor]) + '.png'))

    # def test(self):
    #     self.load_ckpt(qnetwork)  # load model
    #     for i in range(num_episodes):
    #         ## Reset environment and get first new observation
    #         s = env.reset()[0]  # observation is state, integer 0 ~ 15
    #         rAll = 0
    #         if self.render: env.render()
    #         for j in range(99):  # step index, maximum step is 99
    #             ## Choose an action by greedily (with e chance of random action) from the Q-network
    #             allQ = qnetwork(np.asarray([self.to_one_hot(s, 16)], dtype=np.float32)).numpy()
    #             a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

    #             ## Get new state and reward from environment
    #             s1, r, d, _ = env.step(a[0])
    #             rAll += r
    #             s = s1
    #             if self.render: env.render()
    #             ## Reduce chance of random action if an episode is done.
    #             if d: break

    #         print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
    #               .format(i, num_episodes, rAll, time.time() - t0))
    #         self.rList.append(rAll)
    #     print("Correct rate: " + str(sum(self.rList) / num_episodes * 100) + "%")

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    if args.train:
        random_map = generate_random_map(size=10, p=0.3)    
        env = gym.make("FrozenLake-v1", desc=random_map, render_mode="rgb_array")
        env.reset()

        agent = DQNAgent(num_episodes=args.ep, eps_decay=args.decay, discount_factor=args.gamma, learning_rate=args.lr, env=env)
        print('gamma: ', args.gamma, "learning rate: ", args.lr, "eps decay: ", args.decay)

        print(env.desc)

        agent.train()
    
    # if args.test:
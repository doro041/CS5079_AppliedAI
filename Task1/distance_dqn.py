import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from gym.envs.toy_text.frozen_lake import generate_random_map
import argparse
from datetime import datetime

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
        self.rList = [] 
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

    def to_one_hot(self, i, n_classes=None):
        a = np.zeros(n_classes, 'uint8')
        a[i] = 1
        return a

    def get_model(self, inputs_shape):

        ni = tl.layers.Input(inputs_shape, name='observation')
        nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')(ni)
        return tl.models.Model(inputs=ni, outputs=nn)
    
    def save_ckpt(self, model):  # save trained weights
        path = os.path.join('model', '_'.join(['distance', self.alg_name, str(self.learning_rate), str(self.discount_factor), datetime.now().strftime("%Y%m%d_%H%M%S")]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'dqn_model.hdf5'), model)

        with open(os.path.join(path, 'env_desc.txt'), 'w') as f:
            f.write(str(self.env.desc))

    def train(self):
        flattened_map = self.env.desc.flatten()
        all_episode_reward = []
        for i in range(self.num_episodes):
            visited = set()
            s = self.env.reset()[0]
            rAll = 0
            while True:
                allQ = self.qnetwork(np.asarray([self.to_one_hot(s, 100)], dtype=np.float32)).numpy()
                self.q_table[s] = allQ
                a = np.argmax(allQ, 1)

                if np.random.rand(1) < self.eps:
                    a[0] = self.env.action_space.sample()
               
                next_state, r, d, _, _ = self.env.step(a[0])

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
                    r = 1000
                    self.goal_reached_n += 1
                elif flattened_map[next_state] == b'F':
                    r = -1

                if r < 0:
                    r *= dist_reward
                
                q1 = self.qnetwork(np.asarray([self.to_one_hot(next_state, 100)], dtype=np.float32)).numpy()

                max_q1 = np.max(q1)  
                targetQ = allQ
                targetQ[0, a[0]] = r + self.discount_factor * max_q1

                with tf.GradientTape() as tape:
                    _qvalues = self.qnetwork(np.asarray([self.to_one_hot(s, 100)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, self.train_weights)
                self.optimizer.apply_gradients(zip(grad, self.train_weights))

                rAll += r
                s = next_state
                
                if s not in visited:
                    visited.add(s)
                if d == True:
                    if self.eps > self.min_eps:
                        self.eps *= self.eps_decay
                    break

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
        plt.title('Training Reward of the DQN Agent Over the Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['distance', self.alg_name, str(self.learning_rate), str(self.discount_factor), datetime.now().strftime("%Y%m%d_%H%M%S")]) + '.png'))

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    if args.train:
        random_map = generate_random_map(size=10, p=0.3)    

        # The environment commented below was used to test all the agents on the same environment
        # and record the video.

        # random_map = ['SFFFFHHHHH',
        # 'FHHFFFHFFH',
        # 'HHHHHFHHHH',
        # 'HHFHHFFFFH',
        # 'FHFHFFHFFF',
        # 'FFHHFFFHHH',
        # 'HHHHHFFFFH',
        # 'HHHHHHHFHH',
        # 'HFHHHFHFHH',
        # 'HFHHHHHFFG']

        env = gym.make("FrozenLake-v1", desc=random_map, render_mode="rgb_array")
        env.reset()

        agent = DQNAgent(num_episodes=args.ep, eps_decay=args.decay, discount_factor=args.gamma, learning_rate=args.lr, env=env)
        print('___DISTANCE__MAP_')
        print('gamma: ', args.gamma, "learning rate: ", args.lr, "eps decay: ", args.decay)

        print(env.desc)

        agent.train()

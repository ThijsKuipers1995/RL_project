import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

from .models import q_learning, double_q_learning, EpsilonGreedyPolicy, DynamicEpsilonGreedyPolicy
from .environments import *
from .models.utils import running_mean, tqdm

SHAPE = (7,7)
result_target = True
EPSILON = 1

def episode_lengths_q_learning_determinstic(epsilon=EPSILON, num_episodes=3000, n=50, result_target=result_target):
    env = GridworldEnv(shape=SHAPE)
    Q = np.zeros((env.nS, env.nA))
    policy = DynamicEpsilonGreedyPolicy(Q, epsilon=epsilon)

    Q, (episode_lengths, R, _) = q_learning(env, policy, Q, num_episodes)
    plt.imshow(np.argmax(Q, 1).reshape(*SHAPE))
    cbar = plt.colorbar(ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    plt.show()
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths Q-learning (deterministic)")
    plt.show()


def episode_lengths_q_learning_stochastic(epsilon=EPSILON, num_episodes=5000, n=50, result_target=result_target):
    env = GridworldStochasticEnv(shape=SHAPE)
    Q = np.zeros((env.nS, env.nA))
    policy = DynamicEpsilonGreedyPolicy(Q, epsilon=epsilon)

    Q, (episode_lengths, R, _) = q_learning(env, policy, Q, num_episodes)
    plt.imshow(np.argmax(Q, 1).reshape(*SHAPE) % 4)
    cbar = plt.colorbar(ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    plt.show()
    plt.plot(running_mean(R, n))
    plt.title("Episode lengths Q-learning (stochastic)")
    plt.show()


def episode_lengths_double_q_learning_stochastic(epsilon=EPSILON, num_episodes=5000, n=50, result_target=result_target):
    env = GridworldStochasticEnv(shape=SHAPE)
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = DynamicEpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)

    Q, (episode_lengths, R, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.imshow(np.argmax(Q[0], 1).reshape(*SHAPE) % 4)
    cbar = plt.colorbar(ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    plt.show()
    plt.plot(running_mean(R, n))
    plt.title("Episode lengths double Q-learning (stochastic)")
    plt.show()


def episode_lengths_double_q_learning_deterministic(epsilon=EPSILON, num_episodes=3000, n=50, result_target=result_target):
    env = GridworldEnv(shape=SHAPE)
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = DynamicEpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)

    Q, (episode_lengths, R, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.imshow(np.argmax(Q[0], 1).reshape(*SHAPE))
    cbar = plt.colorbar(ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(['UP', 'RIGHT', 'DOWN', 'LEFT'])
    plt.show()
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths double Q-learning (deterministic)")
    plt.show()


def episode_lengths_q_learning_left_right(epsilon=0.1, num_episodes=200, n=5, mu=-0.1):
    env = LeftRightEnv(mu=mu)
    Q = np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)

    _, (episode_lengths, _, _) = q_learning(env, policy, Q, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths Q-learning (Left-Right)")
    plt.show()

def episode_lengths_double_q_learning_left_right(epsilon=0.1, num_episodes=200, n=5, mu=-0.1):
    env = LeftRightEnv(mu=mu)
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)
    Q, (episode_lengths, _, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths double Q-learning (Left-Right)")
    plt.show()


def ratio_left_right(epsilon=0.1, num_episodes=500, mu=-0.1, nr_iters=1000, result_target=True):
    env = LeftRightEnv(mu=mu)
    q_actions, dbl_q_actions = np.zeros(shape=(nr_iters, num_episodes)), np.zeros(shape=(nr_iters, num_episodes))
    q_rewards, dbl_q_rewards = np.zeros(shape=(nr_iters, num_episodes)), np.zeros(shape=(nr_iters, num_episodes))
    # Rs1, Rs2 = [], []
    for i in tqdm(range(nr_iters)):
        Q = np.zeros((env.nS, env.nA))
        policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)
        Q, (_, R, actions) = q_learning(env, policy, Q, num_episodes, verbatim=False, result_target=result_target)
        # print(np.sum(R)/np.count_nonzero(R),np.count_nonzero(R))
        q_rewards[i] = np.array(R)
        q_actions[i] = np.array([action[0] for action in actions])

    for i in tqdm(range(nr_iters)):
        Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
        policy = EpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)
        Q, (_, R, actions) = double_q_learning(env, policy, Qt, Qb, num_episodes, verbatim=False, result_target=result_target)
        # print(np.sum(R)/np.count_nonzero(R), np.count_nonzero(R))
        dbl_q_rewards[i] = np.array(R)
        dbl_q_actions[i] = np.array([action[0] for action in actions])


    q_ratios, dbl_q_ratios = np.mean(q_actions, axis=0), np.mean(dbl_q_actions, axis=0)
    plt.plot(q_ratios, label='Q Learning')
    plt.plot(dbl_q_ratios, label='Double Q Learning')
    plt.legend()
    plt.show()
    q_rewards, dbl_q_rewards = np.mean(q_rewards, axis=0), np.mean(dbl_q_rewards, axis=0)
    plt.plot(q_rewards, label='Q Learning')
    plt.plot(dbl_q_rewards, label='Double Q Learning')
    plt.legend()
    plt.show()


def check_environment_rewards(mu=0.5, num_episodes=1000, n_left_action=10):
    env = LeftRightEnv(mu=mu, n_left_actions=n_left_action)
    env.reset()

    avg_rewards = defaultdict(int)
    action_counts = defaultdict(int)

    for _ in tqdm(range(num_episodes)):
        env.step(1)
        action = np.random.randint(0, n_left_action)
        _, reward, *_ = env.step(action)
        avg_rewards[action] += reward
        action_counts[action] += 1

    print(f"average reward per action from env: {mu}:")

    for action in range(n_left_action):
        avg_reward = avg_rewards[action] / action_counts[action]
        observed = action_counts[action] / num_episodes
        print(f"{action}: {avg_reward:.3f} ({observed:.3f}))")

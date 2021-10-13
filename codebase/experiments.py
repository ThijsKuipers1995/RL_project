import numpy as np
from matplotlib import pyplot as plt
from .models import q_learning, double_q_learning, EpsilonGreedyPolicy
from .environments import GridworldStochasticEnv
from .models.utils import running_mean


def episode_lengths_q_learning(epsilon=0.1, num_episodes=1000, n=50):
    n = 50
    epsilon = epsilon
    num_episodes = 1000
    env = GridworldStochasticEnv()
    Q = np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)

    _, (episode_lengths, _) = double_q_learning(env, policy, Q, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths Q-learning")
    plt.show()


def episode_lengths_q_learning_deterministic():
    n = 50
    epsilon = 0.8
    num_episodes = 1000
    env = WindyGridworldEnv()
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)

    _, (episode_lengths, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths double Q-learning (deterministic)")
    plt.show()

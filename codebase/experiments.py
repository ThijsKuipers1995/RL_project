import numpy as np
from matplotlib import pyplot as plt
from .models import q_learning, double_q_learning, EpsilonGreedyPolicy
from .environments import *
from .models.utils import running_mean


def episode_lengths_q_learning_determinstic(epsilon=0.1, num_episodes=1000, n=50):
    env = GridworldEnv()
    Q = np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)

    _, (episode_lengths, _) = q_learning(env, policy, Q, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths Q-learning (deterministic)")
    plt.show()


def episode_lengths_q_learning_stochastic(epsilon=0.1, num_episodes=1000, n=50):
    env = GridworldStochasticEnv()
    Q = np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon)

    _, (episode_lengths, _) = q_learning(env, policy, Q, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths Q-learning (deterministic)")
    plt.show()


def episode_lengths_double_q_learning_stochastic(epsilon=0.1, num_episodes=1000, n=50):
    env = GridworldStochasticEnv()
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)

    _, (episode_lengths, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths double Q-learning (stochastic)")
    plt.show()


def episode_lengths_double_q_learning_deterministic(epsilon=0.1, num_episodes=1000, n=50):
    env = GridworldEnv()
    Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
    policy = EpsilonGreedyPolicy(Qt + Qb, epsilon=epsilon)

    _, (episode_lengths, _) = double_q_learning(env, policy, Qt, Qb, num_episodes)
    plt.plot(running_mean(episode_lengths, n))
    plt.title("Episode lengths double Q-learning (stochastic)")
    plt.show()

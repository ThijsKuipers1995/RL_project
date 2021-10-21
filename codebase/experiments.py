import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

from .models import (
    q_learning,
    double_q_learning,
    EpsilonGreedyPolicy,
    DynamicEpsilonGreedyPolicy,
    SoftmaxPolicy,
)
from .environments import *
from .models.utils import running_mean, tqdm

SHAPE = (4, 7)


def ratio_left_right(
    epsilon=1,
    num_episodes=500,
    mu=-0.5,
    num_iters=1000,
    result_target=True,
    policy_type="dynamic",
    avg_Q=False,
):
    """
    Visualizes the average ratio of the left action being chosen in the
    starting state of the left-right problem on both Q-learning and double
    Q-learning using a certain policy.

    Inputs:
        epsilon: exploration factor to use for the behavior policy
        num_episodes: number of episodes to run the learning algorithms for.
        mu: Gaussian mean for the action rewards of the left actions.
        num_iters: number of experiments to run and take average results over.
        results_target: Rewards will be evaluated on greedy target policy
                        if True, else on behavior policy
        policy_type: Type of behavior policy to use.
        avg_Q: Visualizes the average Q-value for the left action if set to True
    """
    env = LeftRightEnv(mu=mu)
    q_actions = np.zeros(shape=(num_iters, num_episodes))
    dbl_q_actions = np.zeros(shape=(num_iters, num_episodes))
    delta_q = np.zeros(shape=(num_iters, num_episodes))
    delta_dbl_q = np.zeros(shape=(num_iters, num_episodes))

    if policy_type == "dynamic":
        args = {"epsilon": epsilon}
        policy_cl = DynamicEpsilonGreedyPolicy
    elif policy_type == "epsilon_greedy":
        args = {"epsilon": epsilon}
        policy_cl = EpsilonGreedyPolicy
    elif policy_type == "softmax":
        args = {}
        policy_cl = SoftmaxPolicy
    else:
        raise AssertionError("Policy does not exist")

    for i in tqdm(range(num_iters)):
        Q = np.zeros((env.nS, env.nA))
        policy = policy_cl(Q, **args)
        Qs, (_, _, actions) = q_learning(
            env,
            policy,
            Q,
            num_episodes,
            verbatim=False,
            result_target=result_target,
            all_Q=True,
        )
        q_actions[i] = np.array([action[0] for action in actions])
        delta_q[i] = np.array([(Q[0, 1]) for Q in Qs])

    for i in tqdm(range(num_iters)):
        Qt, Qb = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
        policy = policy_cl(Qt + Qb, **args)
        Qs, (_, _, actions) = double_q_learning(
            env,
            policy,
            Qt,
            Qb,
            num_episodes,
            verbatim=False,
            result_target=result_target,
            all_Q=True,
        )

        dbl_q_actions[i] = np.array([action[0] for action in actions])
        delta_dbl_q[i] = np.array([Q[0, 1] for Q, _ in Qs])

    q_ratios, dbl_q_ratios = np.mean(q_actions, axis=0), np.mean(dbl_q_actions, axis=0)
    plt.plot(q_ratios, label="Q Learning")
    plt.plot(dbl_q_ratios, label="Double Q Learning")
    plt.ylabel("Ratio of left action being taken")
    plt.xlabel("Episode")
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, num_episodes))
    plt.show()

    if avg_Q:
        avg_d_q, avg_dbl_q = np.mean(delta_q, axis=0), np.mean(delta_dbl_q, axis=0)
        q_std, dbl_q_std = np.std(delta_q, axis=0), np.std(delta_dbl_q, axis=0)
        plt.plot(avg_d_q, label="Q Learning")
        plt.fill_between(
            range(num_episodes), avg_d_q - q_std, avg_d_q + q_std, alpha=0.3
        )
        plt.plot(avg_dbl_q, label="Double Q Learning")
        plt.fill_between(
            range(num_episodes), avg_dbl_q - dbl_q_std, avg_dbl_q + dbl_q_std, alpha=0.3
        )
        plt.xlim((0, num_episodes))
        plt.ylabel("Average left action Q-value")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(axis="y")
        plt.show()


def gridworld_test(
    num_iters=100,
    epsilon=1,
    num_episodes=2000,
    n=10,
    result_target=True,
    env_type="left_right_grid",
    policy_type="dynamic",
    length_lim=(0, 60),
    reward_lim=(-200, 0),
):
    """
    Visualizes the episode length and rewards of a gridworld environment using
    a provided behavior policy averaged over num_iters runs.

    Inputs:
        num_iters: number of experiments to run and take average results over.
        epsilon: exploration factor to use for the behavior policy
        num_episodes: number of episodes to run the learning algorithms for.
        n: number of samples to compute the average over for smoothing of plots
        results_target: Rewards will be evaluated on greedy target policy
                        if True, else on behavior policy
        env_type: Type of gridworld environment to use.
        policy_type: Type of behavior policy to use.
        length_lim: y-axis limits to use for the episode lengths plot
        reward_lim: y-axis limits to use for the episode rewards plot
    """

    if env_type == "stochastic":
        env = GridworldStochasticEnv(shape=SHAPE)
    elif env_type == "deterministic":
        env = GridworldEnv(shape=SHAPE)
    elif env_type == "reverse_stochastic":
        env = GridworldReverseStochasticEnv(shape=SHAPE, n_paths=1)
    elif env_type == "left_right_grid":
        env = GridworldLeftRightEnv(shape=SHAPE)
    else:
        raise AssertionError("Environment type does not exist")

    if policy_type == "dynamic":
        pol_args = {"epsilon": epsilon}
        policy_cl = DynamicEpsilonGreedyPolicy
    elif policy_type == "epsilon_greedy":
        pol_args = {"epsilon": epsilon}
        policy_cl = EpsilonGreedyPolicy
    elif policy_type == "softmax":
        pol_args = {}
        policy_cl = SoftmaxPolicy
    else:
        raise AssertionError("Policy does not exist")

    lengths_dbl_q = np.zeros(shape=(num_iters, num_episodes - n))
    rewards_dbl_q = np.zeros(shape=(num_iters, num_episodes - n))
    lengths_q = np.zeros(shape=(num_iters, num_episodes - n))
    rewards_q = np.zeros(shape=(num_iters, num_episodes - n))

    for i in tqdm(range(num_iters)):
        Q = np.zeros((env.nS, env.nA))
        policy = policy_cl(Q, **pol_args)

        Q, (episode_lengths, R, _) = q_learning(
            env, policy, Q, num_episodes, verbatim=False, result_target=result_target
        )

        lengths_q[i] = running_mean(episode_lengths, n)
        rewards_q[i] = running_mean(R, n)

    for i in tqdm(range(num_iters)):
        Q1, Q2 = np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))
        policy = policy_cl(Q1 + Q2, **pol_args)

        (Q1, Q2), (episode_lengths, R, _) = double_q_learning(
            env,
            policy,
            Q1,
            Q2,
            num_episodes,
            verbatim=False,
            result_target=result_target,
        )

        lengths_dbl_q[i] = running_mean(episode_lengths, n)
        rewards_dbl_q[i] = running_mean(R, n)

    avg_double_lengths, avg_double_rewards = np.mean(lengths_dbl_q, axis=0), np.mean(
        rewards_dbl_q, axis=0
    )
    double_length_std, double_reward_std = np.std(lengths_dbl_q, axis=0), np.std(
        rewards_dbl_q, axis=0
    )

    avg_q_lengths, avg_q_rewards = np.mean(lengths_q, axis=0), np.mean(
        rewards_q, axis=0
    )
    q_length_std, q_reward_std = np.std(lengths_q, axis=0), np.std(rewards_q, axis=0)

    plt.plot(avg_q_lengths, label="Q Learning")
    plt.fill_between(
        range(num_episodes - n),
        avg_q_lengths - q_length_std,
        avg_q_lengths + q_length_std,
        alpha=0.3,
    )
    plt.plot(avg_double_lengths, label="Double Q Learning")
    plt.fill_between(
        range(num_episodes - n),
        avg_double_lengths - double_length_std,
        avg_double_lengths + double_length_std,
        alpha=0.3,
    )
    plt.ylim(length_lim)
    plt.legend()
    plt.ylabel("Episode Lengths")
    plt.xlabel("Episode")
    plt.xlim((0, num_episodes))
    plt.show()

    plt.plot(avg_q_rewards, label="Q Learning")
    plt.fill_between(
        range(num_episodes - n),
        avg_q_rewards - q_reward_std,
        avg_q_rewards + q_reward_std,
        alpha=0.3,
    )
    plt.plot(avg_double_rewards, label="Double Q Learning")
    plt.fill_between(
        range(num_episodes - n),
        avg_double_rewards - double_reward_std,
        avg_double_rewards + double_reward_std,
        alpha=0.3,
    )
    plt.legend()
    plt.ylim((reward_lim))
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.xlim((0, num_episodes))
    plt.show()


def run_gridworld(
    policy_type,
    double_Q=False,
    env_type="left_right_grid",
    num_iters=100,
    num_episodes=2000,
    n=10,
):
    """
    Computes and returns the average episode lengths and rewards along with
    their standard deviation for Q learning and double Q learning on a certain
    behavior policy.

    Inputs:
        policy_type: Type of behavior policy to use.
        double_Q: Uses double Q learning if set to True, else False
        env_type: Type of gridworld environment to use.
        num_iters: number of experiments to run and take average results over.
        num_episodes: number of episodes to run the learning algorithms for.
        n: number of samples to compute the average over for smoothing of plots

    Returns:
        avg_lengths: Average number of lengths
        lengths_std: standard deviation of the lengths
        avg_rewards: Average rewards
        rewards_std: standard deviation of rewards
    """

    if env_type == "stochastic":
        env = GridworldStochasticEnv(shape=SHAPE)
    elif env_type == "deterministic":
        env = GridworldEnv(shape=SHAPE)
    elif env_type == "reverse_stochastic":
        env = GridworldReverseStochasticEnv(shape=SHAPE, n_paths=1)
    elif env_type == "left_right_grid":
        env = GridworldLeftRightEnv(shape=SHAPE)
    else:
        raise AssertionError("Environment type does not exist")

    if policy_type == "dynamic":
        pol_args = {"epsilon": 1}
        policy_cl = DynamicEpsilonGreedyPolicy
    elif policy_type == "epsilon_greedy":
        pol_args = {"epsilon": 0.1}
        policy_cl = EpsilonGreedyPolicy
    elif policy_type == "random":
        pol_args = {"epsilon": 0.9}
        policy_cl = EpsilonGreedyPolicy
    elif policy_type == "softmax":
        pol_args = {}
        policy_cl = SoftmaxPolicy
    else:
        raise AssertionError("Policy does not exist")

    lengths = np.zeros(shape=(num_iters, num_episodes - n))
    rewards = np.zeros(shape=(num_iters, num_episodes - n))

    for i in tqdm(range(num_iters)):
        if double_Q:
            Q = [np.zeros((env.nS, env.nA)), np.zeros((env.nS, env.nA))]
            learn_fn = double_q_learning
            policy = policy_cl(Q[0] + Q[1], **pol_args)
        else:
            Q = [np.zeros((env.nS, env.nA))]
            learn_fn = q_learning
            policy = policy_cl(*Q, **pol_args)

        _, (episode_lengths, R, _) = learn_fn(
            env, policy, *Q, num_episodes, verbatim=False
        )

        lengths[i] = running_mean(episode_lengths, n)
        rewards[i] = running_mean(R, n)

    avg_lengths, lengths_std = np.mean(lengths, axis=0), np.std(lengths, axis=0)
    avg_rewards, rewards_std = np.mean(rewards, axis=0), np.std(rewards, axis=0)

    return avg_lengths, lengths_std, avg_rewards, rewards_std


def gridworld_behavior_test(num_iters=100, num_episodes=2000):
    """
    Visualizes the average episode rewards and lengths along with their
    standard deviation for Q Learning and Double Q learning using
    dynamic, epsilon_greedy and softmax behavior policies.

    Inputs:
        num_iters: number of iterations to take average over
        num_episodes: Number of episodes to learn.
    """
    policies = ["dynamic", "epsilon_greedy", "softmax"]
    label_map = {
        "dynamic": "Baseline",
        "epsilon_greedy": "$\epsilon$-greedy ($\epsilon=0.1$)",
        "random": "Random ($\epsilon=0.5$)",
        "softmax": "Softmax",
    }
    results_Q = {policy: {} for policy in policies}
    results_dbl_Q = {policy: {} for policy in policies}

    for policy_type in policies:
        mean_lengths, lengths_std, R_mean, R_std = run_gridworld(
            policy_type, num_iters=num_iters, num_episodes=num_episodes
        )
        results_Q[policy_type]["R"], results_Q[policy_type]["R_std"] = R_mean, R_std
        plt.plot(mean_lengths, label=label_map[policy_type])
        plt.fill_between(
            range(len(mean_lengths)),
            mean_lengths - lengths_std,
            mean_lengths + lengths_std,
            alpha=0.3,
        )
    plt.xlim((0, num_episodes))
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.ylim((0, 60))
    plt.show()

    for policy_type in policies:
        Rs, Rs_std = results_Q[policy_type]["R"], results_Q[policy_type]["R_std"]
        plt.plot(Rs, label=label_map[policy_type])
        plt.fill_between(range(len(Rs)), Rs - Rs_std, Rs + Rs_std, alpha=0.3)

    plt.xlim((0, num_episodes))
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.ylim((-200, 0))
    plt.show()

    for policy_type in policies:
        mean_lengths, lengths_std, R_mean, R_std = run_gridworld(
            policy_type, double_Q=True, num_iters=num_iters, num_episodes=num_episodes
        )
        results_dbl_Q[policy_type]["R"], results_dbl_Q[policy_type]["R_std"] = (
            R_mean,
            R_std,
        )
        plt.plot(mean_lengths, label=label_map[policy_type])
        plt.fill_between(
            range(len(mean_lengths)),
            mean_lengths - lengths_std,
            mean_lengths + lengths_std,
            alpha=0.3,
        )
    plt.xlim((0, num_episodes))
    plt.legend()
    plt.ylim((0, 60))
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.show()

    for policy_type in policies:
        Rs, Rs_std = (
            results_dbl_Q[policy_type]["R"],
            results_dbl_Q[policy_type]["R_std"],
        )
        plt.plot(Rs, label=label_map[policy_type])
        plt.fill_between(range(len(Rs)), Rs - Rs_std, Rs + Rs_std, alpha=0.3)

    plt.xlim((0, num_episodes))
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.ylim((-200, 0))
    plt.show()

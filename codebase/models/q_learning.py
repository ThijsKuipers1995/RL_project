import numpy as np
from .utils import tqdm
from .policies import GreedyPolicy
from collections import defaultdict
from copy import deepcopy


def evaluate(env, target_policy, Q):
    """
    Evaluates learned Q values on an environment using a target policy.
    Inputs:
        env: Gym environment class
        target_policy: Policy class of the target policy
        Q: Learned Q values
    Outputs: tuple(int, int, list)
        i: Length of the episode
        R: Accumulated reward throughout the episode.
        actions: list of the actions taken throughout the episode.
    """
    target_policy.set_Q(Q)
    i = R = 0
    state, done = env.reset(), False
    actions = []
    while not done and i < 100:
        a = target_policy.sample_action(state)
        actions.append(a)
        state, reward, done, _ = env.step(a)
        i, R = i + 1, R + reward
    return i, R, actions


def q_learning(
    env,
    policy,
    Q,
    num_episodes,
    discount_factor=1.0,
    alpha=1.0,
    result_target=True,
    all_Q=False,
    verbatim=True,
):
    """
    Performs Q learning to estimate the Q-values of an environment.
    Inputs:
        env: Gym environment object
        policy: Policy behavior policy object.
        Q: Initial Q_values
        num_episodes: Number of episodes to train on
        discount_factor: Discount_factor of the update step
        alpha: learning rate of the update step
        results_target: Rewards will be evaluated on greedy target policy
                        if True, else on behavior policy
        all_Q: returns a list of all the Q_values for every episode if set
                to True, else only the final learned Q_values.
        verbatim: Prints progress if set to True.

    Returns:
        Q: Q_values (dependant on all_Q flag).
        i: list of episode lengths for each episode
        R: list of accumulated rewards for each episode
        actions: list of lists with all the actions for each episode.
    """
    visisted = defaultdict(int)
    Qs = []
    stats = []

    if result_target:
        target_policy = GreedyPolicy(Q)

    for _ in tqdm(range(num_episodes), disable=not verbatim):
        i = R = 0
        state, done, actions = env.reset(), False, []
        while not done:
            action = policy.sample_action(state, env)
            actions.append(action)
            old_state, state, reward, done, _ = state, *env.step(action)
            visisted[old_state, action] += 1
            Q[old_state, action] += (
                alpha
                / visisted[old_state, action] ** 0.5
                * (reward + discount_factor * max(Q[state]) - Q[old_state, action])
            )
            i, R = i + 1, R + reward

        if result_target:
            _, R, actions = evaluate(env, target_policy, Q)
        if all_Q:
            Qs.append(deepcopy(Q))

        stats.append((i, R, actions))
    if all_Q:
        Q = Qs
    return Q, tuple(zip(*stats))


def double_q_learning(
    env,
    policy,
    Qt,
    Qb,
    num_episodes,
    discount_factor=1.0,
    alpha=1.0,
    random_Q_choice=True,
    eps=0.5,
    update_Q_fn=lambda Qt, Qb: Qt + Qb,
    result_target=True,
    all_Q=False,
    verbatim=True,
):
    """
    Performs Double Q learning to estimate the Q-values of an environment.
    Inputs:
        env: Gym environment object
        policy: Policy behavior policy object.
        Qt: Initial Q-values of the first set.
        Qb: Initial Q-values of the second set.
        num_episodes: Number of episodes to train on
        discount_factor: Discount_factor of the update step
        alpha: learning rate of the update step
        random_Q_choice: Randomly chooses which Q-value set to update if True,
                        updates it one after the other if False.
        update_Q_fn: Function which determines how the two Q-value sets
                    are combined for the action selection.
        results_target: Rewards will be evaluated on greedy target policy
                        if True, else on behavior policy
        all_Q: returns a list of all the Q_values for every episode if set
                to True, else only the final learned Q_values.
        verbatim: Prints progress if set to True.

    Returns:
        Q: Q_values (dependant on all_Q flag).
        i: list of episode lengths for each episode
        R: list of accumulated rewards for each episode
        actions: list of lists with all the actions for each episode.
    """

    visisted_t = defaultdict(int)
    visisted_b = defaultdict(int)
    Qs = []
    stats = []

    if result_target:
        target_policy = GreedyPolicy(update_Q_fn(Qt, Qb))
    # choose randomly based on eps or switch every iteration
    Q_choice_fn = (
        lambda _, eps: np.random.rand() >= eps
        if random_Q_choice
        else lambda i, _: bool(i % 2)
    )

    for _ in tqdm(range(num_episodes), disable=not verbatim):
        i = R = 0
        state, done, actions = env.reset(), False, []
        while not done:
            action = policy.sample_action(state, env)
            actions.append(action)
            old_state, state, reward, done, _ = state, *env.step(action)
            # Updates only one of the two Q-values sets
            if Q_choice_fn(i, eps):
                visisted_t[old_state, action] += 1
                Qt[old_state, action] += (
                    alpha
                    / visisted_t[old_state, action] ** 0.5
                    * (
                        reward
                        + discount_factor * Qb[state, np.argmax(Qt[state])]
                        - Qt[old_state, action]
                    )
                )
            else:
                visisted_b[old_state, action] += 1
                Qb[old_state, action] += (
                    alpha
                    / visisted_b[old_state, action] ** 0.5
                    * (
                        reward
                        + discount_factor * Qt[state, np.argmax(Qb[state])]
                        - Qb[old_state, action]
                    )
                )

            policy.set_Q(update_Q_fn(Qt, Qb))
            i, R = i + 1, R + reward

        if result_target:
            _, R, actions = evaluate(env, target_policy, update_Q_fn(Qt, Qb))
        if all_Q:
            Qs.append((deepcopy(Qb), deepcopy(Qt)))

        stats.append((i, R, actions))

    Q = Qs if all_Q else (Qt, Qb)
    return Q, tuple(zip(*stats))

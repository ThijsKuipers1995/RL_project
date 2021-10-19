import numpy as np
from .utils import tqdm
from .policies import GreedyPolicy
from collections import defaultdict
from copy import deepcopy

def evaluate(env, target_policy, Q):
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

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=1.0, 
                result_target=True, all_Q=False, verbatim=True):
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
            Q[old_state, action] += alpha / visisted[old_state, action]**0.5 * (reward + discount_factor * max(Q[state]) - Q[old_state, action])
            i, R = i + 1, R + reward

        if result_target:
            _, R, actions = evaluate(env, target_policy, Q)
        if all_Q:
            Qs.append(deepcopy(Q))

        stats.append((i, R, actions))
    if all_Q:
        Q = Qs
    return Q, tuple(zip(*stats))


def double_q_learning(env, policy, Qt, Qb, num_episodes, discount_factor=1.0, 
                        alpha=1., random_Q_choice=True, eps=0.5, 
                        update_Q_fn=lambda Qt, Qb: Qt + Qb, result_target=True,
                        all_Q=False, verbatim=True):
    visisted_t = defaultdict(int)
    visisted_b = defaultdict(int)
    Qs = []
    stats = []

    if result_target:
        target_policy = GreedyPolicy(update_Q_fn(Qt, Qb))
    # choose randomly based on eps or switch every iteration
    Q_choice_fn = lambda _, eps: np.random.rand() >= eps if random_Q_choice else lambda i, _: bool(i % 2)

    for _ in tqdm(range(num_episodes), disable=not verbatim):
        i = R = 0
        state, done, actions = env.reset(), False, []
        while not done:
            action = policy.sample_action(state, env)
            actions.append(action)
            old_state, state, reward, done, _ = state, *env.step(action)

            if Q_choice_fn(i, eps):
                visisted_t[old_state, action] += 1
                Qt[old_state, action] += alpha / visisted_t[old_state, action]**0.5 * (reward + discount_factor * Qb[state, np.argmax(Qt[state])] - Qt[old_state, action])
            else:
                visisted_b[old_state, action] += 1
                Qb[old_state, action] += alpha / visisted_b[old_state, action]**0.5 * (reward + discount_factor * Qt[state, np.argmax(Qb[state])] - Qb[old_state, action])

            policy.set_Q(update_Q_fn(Qt, Qb))
            i, R = i + 1, R + reward

        if result_target:
            _, R, actions = evaluate(env, target_policy, update_Q_fn(Qt, Qb))
        if all_Q:
            Qs.append((deepcopy(Qb), deepcopy(Qt)))

        stats.append((i, R, actions))

    Q = Qs if all_Q else (Qt, Qb)
    return Q, tuple(zip(*stats))

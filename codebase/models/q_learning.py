import numpy as np
from .utils import tqdm
from .policies import GreedyPolicy

def evaluate(env, target_policy, Q):
    target_policy.set_Q(Q)
    i = R = 0
    state, done = env.reset(), False
    while not done and i < 100:
        state, reward, done, _ = env.step(target_policy.sample_action(state))
        i, R = i + 1, R + reward
    return i, R

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.1, result_target=True):
    stats = []
    if result_target:
        target_policy = GreedyPolicy(Q)

    for _ in tqdm(range(num_episodes)):
        i = R = 0
        state, done = env.reset(), False
        while not done:
            action = policy.sample_action(state)
            old_state, state, reward, done, _ = state, *env.step(action)
            Q[old_state, action] += alpha * (reward + discount_factor * max(Q[state]) - Q[old_state, action])
            i, R = i + 1, R + reward

        if result_target:
            i, R = evaluate(env, target_policy, Q)

        stats.append((i, R))
    return Q, tuple(zip(*stats))


def double_q_learning(env, policy, Qt, Qb, num_episodes, discount_factor=1.0, alpha=0.1, random_Q_choice=True, eps=0.5, update_Q_fn=lambda Qt, Qb: Qt + Qb, result_target=True):
    stats = []
    if result_target:
        target_policy = GreedyPolicy(update_Q_fn(Qt, Qb))
    # choose randomly based on eps or switch every iteration
    Q_choice_fn = lambda _, eps: np.random.rand() >= eps if random_Q_choice else lambda i, _: bool(i % 2)

    for _ in tqdm(range(num_episodes)):
        i = R = 0
        state, done = env.reset(), False
        while not done:
            action = policy.sample_action(state)
            old_state, state, reward, done, _ = state, *env.step(action)

            if Q_choice_fn(i, eps):
                Qt[old_state, action] += alpha * (reward + discount_factor * Qb[state, np.argmax(Qt[state])] - Qt[old_state, action])
            else:
                Qb[old_state, action] += alpha * (reward + discount_factor * Qt[state, np.argmax(Qb[state])] - Qb[old_state, action])

            policy.set_Q(update_Q_fn(Qt, Qb))
            i, R = i + 1, R + reward

        if result_target:
            i, R = evaluate(env, target_policy, update_Q_fn(Qt, Qb))

        stats.append((i, R))
    return (Qt, Qb), tuple(zip(*stats))

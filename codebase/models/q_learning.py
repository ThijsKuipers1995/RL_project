import numpy as np
from .utils import tqdm


def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    stats = []
    
    for _ in tqdm(range(num_episodes)):
        i = R = 0
        state, done = env.reset(), False
        while not done:
            action = policy.sample_action(state)
            old_state, state, reward, done, _ = state, *env.step(action)
            Q[old_state, action] += alpha * (reward + discount_factor * max(Q[state]) - Q[old_state, action])
            i, R = i + 1, R + reward
        stats.append((i, R))
    return Q, tuple(zip(*stats))


def double_q_learning(env, policy, Qt, Qb, num_episodes, discount_factor=1.0, alpha=0.5, random_Q_choice=False, eps=0.5, update_Q_fn=lambda Qt, Qb: Qt + Qb):
    stats = []

    # choose randomly based on eps or switch every iteration
    Q_choice_fn = lambda _, eps: np.random.rand() >= eps if random_Q_choice else lambda i, _: bool(i % 2) 
    
    for _ in tqdm(range(num_episodes)):
        i = R = 0
        state, done = env.reset(), False
        while not done:
            action = policy.sample_action(state)
            old_state, state, reward, done, _ = state, *env.step(action)
            
            if Q_choice_fn(i, eps):
                Qt[old_state, action] += alpha * (reward + discount_factor * Qb[state, np.argmax(Qt[state])]) - Qt[old_state, action]
            else:
                Qb[old_state, action] += alpha * (reward + discount_factor * Qt[state, np.argmax(Qb[state])]) - Qb[old_state, action]
            
            policy.set_Q(update_Q_fn(Qt, Qb))
            i, R = i + 1, R + reward
        stats.append((i, R))
    return tuple(Qt, Qb), tuple(zip(*stats))

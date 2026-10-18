"""Tabular Q-learning, implemented from scratch for Gymnasium Taxi-v4."""
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

ENV_ID = "Taxi-v4"


@dataclass
class TrainConfig:
    episodes: int = 20_000
    alpha: float = 0.7
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_fraction: float = 0.80
    seed: int = 42


def epsilon_at(episode: int, config: TrainConfig) -> float:
    """Decay epsilon linearly, then keep the exploration floor."""
    fraction = min(1.0, episode / max(1, config.episodes * config.decay_fraction))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def train_q_learning(config: TrainConfig, verbose: bool = True):
    """Learn Q(s,a) without pretrained libraries or environment action masks."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(config.seed)
    q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    rewards, lengths, successes = [], [], []
    log_every = max(1, config.episodes // 10)

    try:
        for ep in range(config.episodes):
            state, _ = env.reset(seed=config.seed + ep)
            epsilon = epsilon_at(ep, config)
            total_reward, episode_length, succeeded = 0.0, 0, False

            while True:
                if rng.random() < epsilon:
                    action = int(rng.integers(env.action_space.n))
                else:
                    best = np.flatnonzero(q[state] == q[state].max())
                    action = int(rng.choice(best))

                next_state, reward, terminated, truncated, _ = env.step(action)
                # Time-limit truncation is NOT an MDP terminal state:
                # bootstrap at truncation; don't bootstrap at termination.
                target = reward if terminated else reward + config.gamma * q[next_state].max()
                q[state, action] += config.alpha * (target - q[state, action])

                total_reward += float(reward)
                episode_length += 1
                state = next_state
                if terminated or truncated:
                    succeeded = bool(terminated)  # Taxi terminates upon successful delivery
                    break

            rewards.append(total_reward)
            lengths.append(episode_length)
            successes.append(int(succeeded))

            if verbose and ((ep + 1) % log_every == 0 or ep == 0):
                window = min(ep + 1, 250)
                print(f"Episode {ep+1:>6}/{config.episodes} | "
                      f"mean reward (last {window}): {np.mean(rewards[-window:]):>7.2f} | "
                      f"success: {100*np.mean(successes[-window:]):>5.1f}% | "
                      f"epsilon: {epsilon:.3f}")
    finally:
        env.close()

    return q, {
        "episode": np.arange(1, config.episodes + 1),
        "reward": np.array(rewards),
        "steps": np.array(lengths),
        "success": np.array(successes),
    }


def evaluate_policy(q_table=None, episodes: int = 100, seed: int = 10_000, random_policy=False):
    """Evaluate with exploration turned off, using the same fixed initial states."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed + 100)
    rewards, lengths, successes = [], [], []
    try:
        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            total_reward, length = 0.0, 0
            while True:
                if random_policy:
                    action = int(rng.integers(env.action_space.n))
                else:
                    action = int(np.argmax(q_table[state]))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    rewards.append(total_reward)
                    lengths.append(length)
                    successes.append(int(terminated))
                    break
    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(lengths)),
        "episodes": int(episodes),
    }

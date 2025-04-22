import numpy as np
import math
import random

class MultiArmedBandit:
    def __init__(self, num_arms=5, algorithm='epsilon-greedy',
                 epsilon=0.1, temperature=0.1, ucb_constant=2):
        self.num_arms = num_arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.temperature = temperature
        self.ucb_constant = ucb_constant
        self.reset()

    def reset(self):
        # True (hidden) probabilities
        self.true_rewards = np.random.rand(self.num_arms)
        # Estimates and counts
        self.estimated_rewards = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms, dtype=int)
        # Tracking
        self.current_trial = 0
        self.cumulative_reward = 0.0
        self.regret_history = []
        self.history = []  # list of dicts for each trial

    def select_arm(self):
        if self.algorithm == 'epsilon-greedy':
            if random.random() < self.epsilon or np.any(self.arm_counts == 0):
                return random.randrange(self.num_arms)
            return int(np.argmax(self.estimated_rewards))

        elif self.algorithm == 'ucb':
            if np.any(self.arm_counts == 0):
                return int(np.argmin(self.arm_counts))
            ucb_values = self.estimated_rewards + \
                np.sqrt((self.ucb_constant * np.log(self.current_trial + 1)) / self.arm_counts)
            return int(np.argmax(ucb_values))

        elif self.algorithm == 'thompson':
            samples = []
            for i in range(self.num_arms):
                # Beta posterior: α = successes+1, β = failures+1
                successes = self.estimated_rewards[i] * self.arm_counts[i]
                failures = self.arm_counts[i] - successes
                a, b = successes + 1, failures + 1
                samples.append(np.random.beta(a, b))
            return int(np.argmax(samples))

        elif self.algorithm == 'softmax':
            if np.any(self.arm_counts == 0):
                return int(np.argmin(self.arm_counts))
            shifted = self.estimated_rewards - np.max(self.estimated_rewards)
            exp_vals = np.exp(shifted / self.temperature)
            probs = exp_vals / np.sum(exp_vals)
            return int(np.random.choice(self.num_arms, p=probs))

        else:
            return random.randrange(self.num_arms)

    def step(self):
        arm = self.select_arm()
        reward = 1 if random.random() < self.true_rewards[arm] else 0

        # Update counts & estimates
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.estimated_rewards[arm] += (reward - self.estimated_rewards[arm]) / n

        # Update tracking
        self.current_trial += 1
        self.cumulative_reward += reward

        # Regret
        optimal = np.max(self.true_rewards)
        regret = optimal - self.true_rewards[arm]
        self.regret_history.append(regret)

        # Record history
        self.history.append({
            'trial': self.current_trial,
            'selected_arm': arm + 1,
            'reward': reward,
            'cumulative_reward': self.cumulative_reward,
            'cumulative_regret': sum(self.regret_history),
            **{f'arm{idx+1}_est': est for idx, est in enumerate(self.estimated_rewards)},
            **{f'arm{idx+1}_count': cnt for idx, cnt in enumerate(self.arm_counts)},
        })
        return self.history[-1]

import numpy as np
from collections import deque, namedtuple
import torch

from sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    Memory buffer responsible for Prioritized Experience Replay.

    This buffer stores up to memory_size experiences in a circular
    array-like data structure.  Each experience is also associated
    with a probability weight.

    Batches may be sampled (with replacement) from this implied
    probability distribution in batches.  The provided weights should
    be non-negative, but are not required to add up to 1.
    """

    def __init__(self, device, memory_size, update_every=4, seed=0):
        """  Initializes the data structure

        :param device:  (torch.device) Object representing the device where to allocate tensors
        :param memory_size: (int) Maximum capacity of memory buffer
        :param update_every: (int) Number of steps between update operations
        :param seed:  (int) Seed used for PRNG
        """
        self.device = device
        self.probability_weights = SumTree(capacity=memory_size, seed=seed)
        self.elements = deque(maxlen=memory_size)
        self.update_every = update_every

        self.step = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """  Adds a experience tuple (s, a, r, s', done) to memory

        :param state:  (array-like)  State value from experience tuple
        :param action:  (int)  Action value from experience tuple
        :param reward:  (float)  Reward value from experience tuple
        :param next_state:  (array-like)  Next state value from experience tuple
        :param done:  (bool)  Done flag from experience tuple
        """
        e = self.experience(state, action, reward, next_state, done)
        self.elements.append(e)
        self.step += 1

        # Add batch of experiences to memory, with max initial weight
        if self.step >= self.update_every:
            self.probability_weights.add(self.step)
            self.step = 0

    def sample(self, batch_size, alpha, beta):
        """  Samples a batch of examples with replacement from the buffer.

        :param batch_size:  (int)  Number of samples to sample
        :param alpha:  (float) PER probability hyperparameter
        :param beta:  (float) PER probability hyperparameter
        :return:
            states:  (list)  States from sampled experiences
            actions:  (list)  Actions from sampled experiences
            rewards:  (list)  Rewards from sampled experiences
            next_states:  (list)  Next states from sampled experiences
            dones:  (list)  Done flags from sampled experiences
            indexes:  (list)  Indexes of sampled experiences
        """
        indexes = self.probability_weights.sample(batch_size=batch_size, alpha=alpha, beta=beta)
        experiences = [self.elements[i] for i in indexes]

        # Copy experience tensors to device
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones, indexes

    def update(self, indexes, weights):
        """  Updates the probability weights associated with the provided indexes.

        :param indexes:  (array indexes) Indexes to have weights updated
        :param weights:  (list) New weights for the provided indexes
        """
        self.probability_weights.update(indexes, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.probability_weights)

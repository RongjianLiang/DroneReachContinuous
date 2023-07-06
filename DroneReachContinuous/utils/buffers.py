from collections import deque
import random


class ReplayBuffer:
    """
    A basic class wrapper around Python deque that implements basic append and random
    sampling functions.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

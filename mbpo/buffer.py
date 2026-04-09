import collections
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        if batch_size > len(self.buffer):
            return self.return_all_samples()
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def return_all_samples(self):
        if len(self.buffer) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

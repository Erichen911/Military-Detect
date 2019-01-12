import torch
import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity, bs):
        self.max_capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.batch_size = bs

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, self.batch_size))
        S = torch.cat(state, dim=0)
        A = torch.LongTensor(action)
        R = torch.FloatTensor(reward)
        S_pr = torch.cat(next_state, dim=0)
        is_done = torch.FloatTensor(done)
        return S, A, R, S_pr, is_done

    def __len__(self):
        return len(self.buffer)

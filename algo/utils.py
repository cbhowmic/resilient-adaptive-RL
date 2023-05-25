import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.DoubleTensor(self.state[ind]).to(self.device),
            torch.DoubleTensor(self.action[ind]).to(self.device),
            torch.DoubleTensor(self.next_state[ind]).to(self.device),
            torch.DoubleTensor(self.reward[ind]).to(self.device),
            torch.DoubleTensor(self.not_done[ind]).to(self.device)
        )


class Rollouts(object):
    def __init__(self):
        self.rollouts = []

    def append(self, transition):
        self.rollouts.append(transition)

    def make_batch(self, device):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.rollouts:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), torch.stack(a_lst).to(
            device), \
                                              torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst,
                                                                                           dtype=torch.float).to(
            device), \
                                              torch.tensor(done_lst, dtype=torch.float).to(device), torch.tensor(
            prob_a_lst).to(device)
        self.rollouts = []
        return s, a, r, s_prime, done_mask, prob_a

    def choose_mini_batch(self, mini_batch_size, states, actions, rewards, next_states, done_mask, old_log_prob,
                          advantages, returns, old_value):
        full_batch_size = len(states)
        full_indices = np.arange(full_batch_size)
        np.random.shuffle(full_indices)
        for i in range(full_batch_size // mini_batch_size):
            indices = full_indices[mini_batch_size * i: mini_batch_size * (i + 1)]
            yield states[indices], actions[indices], rewards[indices], next_states[indices], done_mask[indices], \
                  old_log_prob[indices], advantages[indices], returns[indices], old_value[indices]

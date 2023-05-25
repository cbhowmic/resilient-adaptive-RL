import torch
import torch.nn as nn
import torch.optim as optim
from algo.utils import Rollouts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.pi = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.pi(x)
        std = torch.exp(self.actor_logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.v = nn.Linear(hidden_dim, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.v(x)
        return v

class PPO(object):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, entropy_coef=1e-2, critic_coef=0.5,
                 discount=0.99, lmbda=0.95, eps_clip=0.2, K_epoch=10, minibatch_size=64, device=device):
        super(PPO, self).__init__()

        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.discount = discount
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.minibatch_size = minibatch_size
        self.max_grad_norm = 0.5

        self.data = Rollouts()

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.device = device

    def pi(self, x):
        mu, sigma = self.actor(x)
        return mu, sigma

    def v(self, x):
        return self.critic(x)

    def put_data(self, transition):
        self.data.append(transition)

    def train(self):
        s_, a_, r_, s_prime_, done_mask_, old_log_prob_ = self.data.make_batch(self.device)
        old_value_ = self.v(s_).detach()
        td_target = r_ + self.discount * self.v(s_prime_) * done_mask_
        delta = td_target - old_value_
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if done_mask_[idx] == 0:
                advantage = 0.0
            advantage = self.discount * self.lmbda * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage_ = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        returns_ = advantage_ + old_value_
        advantage_ = (advantage_ - advantage_.mean()) / (advantage_.std() + 1e-3)
        for i in range(self.K_epoch):
            for s, a, r, s_prime, done_mask, old_log_prob, advantage, return_, old_value in self.data.choose_mini_batch( \
                    self.minibatch_size, s_, a_, r_, s_prime_, done_mask_, old_log_prob_, advantage_, returns_,
                    old_value_):
                curr_mu, curr_sigma = self.pi(s)
                value = self.v(s).float()
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                entropy = curr_dist.entropy() * self.entropy_coef
                curr_log_prob = curr_dist.log_prob(a).sum(1, keepdim=True)

                ratio = torch.exp(curr_log_prob - old_log_prob.detach())

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                actor_loss = (-torch.min(surr1, surr2) - entropy).mean()

                old_value_clipped = old_value + (value - old_value).clamp(-self.eps_clip, self.eps_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)

                critic_loss = 0.5 * self.critic_coef * torch.max(value_loss, value_loss_clipped).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

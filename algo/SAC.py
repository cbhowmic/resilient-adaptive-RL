import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from algo.utils import ReplayBuffer
from utils import hard_target_update, soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def identity(x):
    """Return input without any change."""
    return x


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64, 64),
                 activation=F.relu,
                 output_activation=identity,
                 use_output_layer=True,
                 use_actor=False,
                 ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_limit = output_limit
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer
        self.use_actor = use_actor

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        # If the network is used as actor network, make sure output is in correct range
        x = x * self.output_limit if self.use_actor else x
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ReparamGaussianPolicy(MLP):
    def __init__(self,
                 input_size,
                 output_size,
                 output_limit=1.0,
                 hidden_sizes=(64, 64),
                 activation=F.relu,
                 ):
        super(ReparamGaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]
        self.output_limit = output_limit

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x) * clip_up + (l - x) * clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        return mu, pi, log_pi

    def forward(self, x):
        x = super(ReparamGaussianPolicy, self).forward(x)

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)

        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        pi = dist.rsample()  # Reparameterization trick (mean + std * N(0,1))
        log_pi = dist.log_prob(pi).sum(dim=-1)
        mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)

        # Make sure outputs are in correct range
        mu = mu * self.output_limit
        pi = pi * self.output_limit
        return mu, pi, log_pi


class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x, a], dim=-1)
        return super(FlattenMLP, self).forward(q)


class SAC(object):
    """
    An implementation of agents for Soft Actor-Critic (SAC), SAC with automatic entropy adjustment (SAC-AEA).
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 alpha=0.2,
                 automatic_entropy_tuning=False,
                 hidden_sizes=(256, 256),
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 # policy_lr=3e-4,
                 # qf_lr=3e-4,
                 tau=0.005,
                 policy_losses=list(),
                 qf1_losses=list(),
                 qf2_losses=list(),
                 alpha_losses=list(),
                 ):
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.act_limit = max_action
        self.gamma = discount
        self.alpha = alpha
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.hidden_sizes = hidden_sizes
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_losses = policy_losses
        self.qf1_losses = qf1_losses
        self.qf2_losses = qf2_losses
        self.alpha_losses = alpha_losses

        # Main network
        self.policy = ReparamGaussianPolicy(self.obs_dim, self.act_dim, self.act_limit,
                                            hidden_sizes=self.hidden_sizes).to(device)
        self.qf1 = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)
        self.qf2 = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)
        # Target network
        self.qf1_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)
        self.qf2_target = FlattenMLP(self.obs_dim + self.act_dim, 1, hidden_sizes=self.hidden_sizes).to(device)

        # Initialize target parameters to match main parameters
        hard_target_update(self.qf1, self.qf1_target)
        hard_target_update(self.qf2, self.qf2_target)

        # Concat the Q-network parameters to use one optim
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=self.qf_lr)

        # If automatic entropy tuning is True,
        # initialize a target entropy, a log alpha and an alpha optimizer
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((self.act_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.policy_lr)

    def train(self, replay_buffer, batch_size=256):
        # batch = replay_buffer.sample(batch_size)
        # obs1 = batch['state']
        # obs2 = batch['next_state']
        # acts = batch['action']
        # rews = batch['reward']
        # not_done = batch['not_done']

        obs1, acts, obs2, rews, not_done = replay_buffer.sample(batch_size)
        rews, not_done = rews.squeeze(1), not_done.squeeze(1)

        # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
        _, pi, log_pi = self.policy(obs1)
        _, next_pi, next_log_pi = self.policy(obs2)
        q1 = self.qf1(obs1, acts).squeeze(1)
        q2 = self.qf2(obs1, acts).squeeze(1)

        # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))), min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
        min_q_pi = torch.min(self.qf1(obs1, pi), self.qf2(obs1, pi)).squeeze(1).to(device)
        min_q_next_pi = torch.min(self.qf1_target(obs2, next_pi),
                                  self.qf2_target(obs2, next_pi)).squeeze(1).to(device)

        # Targets for Q regression
        v_backup = min_q_next_pi - self.alpha * next_log_pi
        q_backup = rews + self.gamma * not_done * v_backup
        q_backup.to(device)

        # SAC losses
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())
        qf_loss = qf1_loss + qf2_loss

        # Update policy network parameter
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update two Q-network parameter
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # If automatic entropy tuning is True, update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # Save alpha loss
            self.alpha_losses.append(alpha_loss.item())

        # Update the frozen target models
        # for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        #
        # for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        soft_update(self.qf1_target, self.qf1)
        soft_update(self.qf2_target, self.qf2)

        # Save losses
        self.policy_losses.append(policy_loss.item())
        self.qf1_losses.append(qf1_loss.item())
        self.qf2_losses.append(qf2_loss.item())

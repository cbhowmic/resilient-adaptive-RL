import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm(policy, critic, critic2, actor, state, eps=0.005):
    state = torch.DoubleTensor(state.reshape(1, -1)).to(device)
    state.requires_grad = True
    if policy == "TD3":
        actor_loss = -critic.Q1(state, actor(state)).to(device)
    elif policy == "DDPG":
        actor_loss = -critic(state, actor(state)).to(device)
    elif policy == "SAC":
        # full strategy
        # alpha = 0.2
        # _, pi, log_pi = actor(state)
        # pi, log_pi = pi.to(device), log_pi.to(device)
        # min_q_pi = torch.min(critic(state, pi), critic2(state, pi)).to(device)
        # actor_loss = (alpha * log_pi - min_q_pi).to(device)
        # critic2.zero_grad()
        # short strategy
        actor_loss = -critic(state, actor(state)[1]).to(device)
    else:
        raise NameError("Policy name is not defined!")
    critic.zero_grad()
    actor.zero_grad()
    actor_loss.backward()
    attack_state = state + eps * state.grad.sign()
    attack_state = torch.clamp(attack_state, 0, 1)

    return attack_state.cpu().detach().numpy()

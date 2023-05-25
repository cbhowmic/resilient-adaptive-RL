from __future__ import division

from copy import deepcopy
import torch
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch

def average_rule(keys, Temp_state_dict, neighbors):
    aggr_state_dict = {}
    # aggr_state_dict= torch.sum(Temp_state_dict, 0)
    for key in keys:
        temp_state_dict = [deepcopy(Temp_state_dict[key][i]) for i in neighbors]
        aggr_state_dict[key] = torch.mean(torch.stack(temp_state_dict), 0)
    return aggr_state_dict


def median_rule(keys, Temp_state_dict, neighbors):
    aggr_state_dict = {}
    for key in keys:
        temp_state_dict = [Temp_state_dict[key][i] for i in neighbors]
        aggr_state_dict[key], _ = torch.median(torch.stack(temp_state_dict), 0)
    return aggr_state_dict


def actor_rule(agent_id, policy, Model_actor, Model_critic, Model_critic_2, ram, keys, ActorDict, neighbors, alpha, Accumu_Q_actor, filter, normalize=False, softmax=False):
    random_batch_size = 256
    # gamma = 1
    s1, a1, s2, _, _ = ram.sample(random_batch_size)
    # s1 = Variable(torch.from_numpy(np.float32(s1))).to(device)

    for neigh in neighbors:
        if policy == "TD3":
            pred_a1 = Model_actor[neigh](s1)
            Q_actor = Model_critic[agent_id].Q1(s1, pred_a1).mean()
            # Accumu_loss_actor[agent_id, neigh] = (1 - gamma) * Accumu_loss_actor[agent_id, neigh] + gamma * loss_actor
            Accumu_Q_actor[agent_id, neigh] = Q_actor
        elif policy == "DDPG":
            pred_a1 = Model_actor[neigh](s1)
            Q_actor = Model_critic[agent_id].forward(s1, pred_a1).mean()
            # Accumu_loss_actor[agent_id, neigh] = (1 - gamma) * Accumu_loss_actor[agent_id, neigh] + gamma * loss_actor
            Accumu_Q_actor[agent_id, neigh] = Q_actor
        elif policy == "PPO":
            pass
        elif policy == "SAC":
            # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
            _, pi, log_pi = Model_actor[neigh](s1)
            # Min Double-Q: min(Q1(s,π(a|s)), Q2(s,π(a|s))), min(Q1‾(s',π(a'|s')), Q2‾(s',π(a'|s')))
            min_q_pi = torch.min(Model_critic[agent_id](s1, pi), Model_critic_2[agent_id](s1, pi)).squeeze(1)
            # SAC losses
            para = 0.2
            policy_loss = (para * log_pi - min_q_pi).mean()
            Accumu_Q_actor[agent_id, neigh] = -policy_loss
        else:
            raise NameError("Policy name is not defined!")

    Q = deepcopy(Accumu_Q_actor[agent_id, :])
    min_Q = np.min(Accumu_Q_actor[agent_id, neighbors])
    max_Q = np.max(Accumu_Q_actor[agent_id, neighbors])

    if normalize:
        # Q = np.array([Q[neigh] - min_Q if neigh in neighbors else 0 for neigh in range(len(Q))])
        # Q = Q / (max_Q - min_Q)
        Q = [Q[neigh] - max_Q if neigh in neighbors else 0 for neigh in range(len(Q))]
        Q = [np.exp(Q[neigh]) if neigh in neighbors else 0 for neigh in range(len(Q))]

    if softmax:
        if not normalize:
            Q = [Q[neigh] - max_Q if neigh in neighbors else 0 for neigh in range(len(Q))]
        Q = [np.exp(Q[neigh]) if neigh in neighbors else 0 for neigh in range(len(Q))]

    if filter:
        Q = [Q[neigh] if Q[neigh] >= Q[agent_id] else 0 for neigh in range(len(Q))]


    Q[agent_id] *= alpha[agent_id]
    sum_Q = sum(Q)
    Weight = Q / sum_Q

    # in case sum is not 1
    Weight[agent_id] = 1 - sum(Weight[:agent_id]) - sum(Weight[agent_id + 1:])
    print("agent %d, actor weight, loss" % agent_id, Weight, Accumu_Q_actor[agent_id, :])

    aggr_state_dict = {}
    for key in keys:
        # temp_state_dict = [ActorDict[key][i] * Weight[i] * len(neighbors) for i in neighbors]
        # aggr_state_dict[key] = torch.mean(torch.stack(temp_state_dict), 0)
        temp_state_dict = [ActorDict[key][i] * Weight[i] for i in neighbors]
        aggr_state_dict[key] = torch.sum(torch.stack(temp_state_dict), 0)

    # filtering
    # aggr_actor = deepcopy(Model_actor[agent_id])
    # aggr_actor.load_state_dict(aggr_state_dict)
    # pred_a1 = aggr_actor(s1)
    # Q_actor = Model_critic[agent_id].Q1(s1, pred_a1).mean()
    # if Q_actor > Accumu_Q_actor[agent_id, agent_id]:
    #     print("agent %d, return aggregate model" % agent_id)
    #     return aggr_state_dict
    # else:
    #     return Model_actor[agent_id].state_dict()

    return aggr_state_dict

def critic_rule(agent_id, policy, Model_actor, Model_critic, Model_critic_2, Model_target_critic, Model_target_critic_2, ram, keys, CriticDict, Critic2Dict, neighbors, alpha, Accumu_loss_critic, filter, softmax=False):
    random_batch_size = 256
    GAMMA = 0.99
    gamma = 1
    s1, a1, s2, r1, not_done = ram.sample(random_batch_size)
    if policy == "SAC":
        r1, not_done = r1.squeeze(1), not_done.squeeze(1)

    for neigh in neighbors:
        # Use target actor exploitation policy here for loss evaluation
        if policy == "TD3":
            a2_k = Model_actor[agent_id](s2).detach()
            target_Q1, target_Q2 = Model_target_critic[agent_id].forward(s2, a2_k)
            target_Q = torch.min(target_Q1, target_Q2)
            # y_exp = r + gamma*Q'( s2, pi'(s2))
            y_expected = r1 + not_done * GAMMA * target_Q
            # y_pred = Q( s1, a1)
            y_predicted_1, y_predicted_2 = Model_critic[neigh].forward(s1, a1)
            # compute critic loss, and update the critic
            loss_critic = F.mse_loss(y_predicted_1, y_expected) + F.mse_loss(y_predicted_2, y_expected)
        elif policy == "DDPG":
            a2_k = Model_actor[agent_id](s2).detach()
            target_Q = Model_target_critic[agent_id].forward(s2, a2_k)
            # y_exp = r + gamma*Q'( s2, pi'(s2))
            y_expected = r1 + not_done * GAMMA * target_Q
            # y_pred = Q( s1, a1)
            y_predicted = Model_critic[neigh].forward(s1, a1)
            # compute critic loss, and update the critic
            loss_critic = F.mse_loss(y_predicted, y_expected)
        elif policy == "PPO":
            pass
        elif policy == "SAC":
            para = 0.2
            # Prediction π(a|s), logπ(a|s), π(a'|s'), logπ(a'|s'), Q1(s,a), Q2(s,a)
            _, next_pi, next_log_pi = Model_actor[agent_id](s2)
            q1 = Model_critic[neigh](s1, a1).squeeze(1)
            q2 = Model_critic_2[neigh](s1, a1).squeeze(1)

            min_q_next_pi = torch.min(Model_target_critic[agent_id](s2, next_pi),
                                      Model_target_critic_2[agent_id](s2, next_pi)).squeeze(1)

            v_backup = min_q_next_pi - para * next_log_pi
            q_backup = r1 + GAMMA * not_done * v_backup

            qf1_loss = F.mse_loss(q1, q_backup.detach())
            qf2_loss = F.mse_loss(q2, q_backup.detach())
            loss_critic = qf1_loss + qf2_loss
        else:
            raise NameError("Policy name is not defined!")
        Accumu_loss_critic[agent_id, neigh] = (1 - gamma) * Accumu_loss_critic[agent_id, neigh] + gamma * loss_critic

    loss = deepcopy(Accumu_loss_critic[agent_id, :])

    # if normalize:
    #     min_Q = np.min(loss)
    #     max_Q = np.max(loss)
    #     loss = (loss - min_Q) / (max_Q - min_Q)

    reversed_Loss = np.zeros(len(Model_actor))
    for neigh in neighbors:
        if filter:
            if Accumu_loss_critic[agent_id, neigh] <= Accumu_loss_critic[agent_id, agent_id]:
                reversed_Loss[neigh] = 1 / loss[neigh]
        else:
            # if softmax:
            #     reversed_Loss[neigh] = np.exp(-loss[neigh])  # 1 / np.exp(loss[neigh])
            # else:
            reversed_Loss[neigh] = 1 / loss[neigh]

    reversed_Loss[agent_id] *= alpha[agent_id]

    sum_reversedLoss = sum(reversed_Loss)
    # Weight = np.zeros(numAgent)
    # for neigh in range(0, numAgent):
    Weight = reversed_Loss / sum_reversedLoss
    # in case sum is not 1
    Weight[agent_id] = 1 - sum(Weight[:agent_id]) - sum(Weight[agent_id + 1:])
    print("agent %d, critic weight, loss, reversedloss" % agent_id, Weight, loss, reversed_Loss)
    # weight = torch.from_numpy(weight)

    aggr_state_dict = {}
    for key in keys:
        # temp_state_dict = [ActorDict[key][i] * Weight[i] * len(neighbors) for i in neighbors]
        # aggr_state_dict[key] = torch.mean(torch.stack(temp_state_dict), 0)
        temp_state_dict = [CriticDict[key][i] * Weight[i] for i in neighbors]
        aggr_state_dict[key] = torch.sum(torch.stack(temp_state_dict), 0)

    if policy == "SAC":
        aggr_state_dict_2 = {}
        for key in keys:
            # temp_state_dict = [ActorDict[key][i] * Weight[i] * len(neighbors) for i in neighbors]
            # aggr_state_dict[key] = torch.mean(torch.stack(temp_state_dict), 0)
            temp_state_dict_2 = [Critic2Dict[key][i] * Weight[i] for i in neighbors]
            aggr_state_dict_2[key] = torch.sum(torch.stack(temp_state_dict_2), 0)
        return aggr_state_dict, aggr_state_dict_2

    return aggr_state_dict



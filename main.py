from __future__ import division

from collections import defaultdict
from threading import Thread
from copy import deepcopy
import torch
import random
import errno
import numpy as np
import gym
import argparse
import os
import time

from utils import random_point_set, findNeighbors, soft_update
from Agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(1)  #OMP_NUM_THREADS=1
torch.set_default_dtype(torch.float64)


def getModelDict(policy, models):
    Model_actor, Model_critic = deepcopy(models["actor"]), deepcopy(models["critic"])
    ActorDict, CriticDict, Critic2Dict = defaultdict(list), defaultdict(list), defaultdict(list)
    Actorlayers, CriticLayers = Model_actor[0].state_dict().keys(), Model_critic[0].state_dict().keys()
    for Model, ModelDict, layers in zip([Model_actor, Model_critic], [ActorDict, CriticDict],
                                        [Actorlayers, CriticLayers]):
        for k in range(numAgent):
            model = Model[k]
            for key in layers:
                if k not in attacker:
                    ModelDict[key].append(model.state_dict()[key])
                else:
                    ModelDict[key].append(model.state_dict()[key] * random.random())
    if policy == "SAC":
        Model_critic_2 = deepcopy(models["critic_2"])
        for k in range(numAgent):
            model = Model_critic_2[k]
            for key in CriticLayers:
                if k not in attacker:
                    Critic2Dict[key].append(model.state_dict()[key])
                else:
                    Critic2Dict[key].append(model.state_dict()[key] * random.random())

    return ActorDict, CriticDict, Critic2Dict, Actorlayers, CriticLayers


def save_reward(Reward, Steps, rule, multi_task, env_name, normalAgents, noise_amp, lr, filter, intruder, fgsm_attack,
                normalize, softmax, policy, actorAggOnly):
    noise_string = "_".join([str(x) for x in noise_amp])
    lr_string = "_".join([str(x) for x in lr])
    task = "multi_task_" if multi_task else ""
    intrud = "_%dintruder" % intruder if intruder > 0 else ""
    fgsm = "_%dfgsm" % fgsm_attack if fgsm_attack > 0 else ""
    if rule == "ResAgg" or rule == "ResAgg_Q":
        rule = rule + "_filter" if filter else rule + "_no_filter"
        rule = rule + "_normalize" if normalize else rule + ""
        rule = rule + "_softmax" if softmax else rule + ""
    actorAggOnly = "_actorAggOnly" if actorAggOnly else ""

    if not os.path.exists("data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s" % (policy, task,
                                                                                                           env_name,
                                                                                                           numAgent,
                                                                                                           attacker_num,
                                                                                                           noise_string,
                                                                                                           lr_string,
                                                                                                           intrud,
                                                                                                           fgsm,
                                                                                                           actorAggOnly) + '/'):
        try:
            os.makedirs("data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s" % (policy, task,
                                                                                                         env_name,
                                                                                                         numAgent,
                                                                                                         attacker_num,
                                                                                                         noise_string,
                                                                                                         lr_string,
                                                                                                         intrud,
                                                                                                         fgsm,
                                                                                                         actorAggOnly) + '/')
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    np.save("data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s/reward_%s.npy" % (policy, task,
                                                                                                           env_name,
                                                                                                           numAgent,
                                                                                                           attacker_num,
                                                                                                           noise_string,
                                                                                                           lr_string,
                                                                                                           intrud, fgsm,
                                                                                                           actorAggOnly,
                                                                                                           rule), Reward)
    np.save("data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s/steps_%s.npy" % (policy, task,
                                                                                                          env_name,
                                                                                                          numAgent,
                                                                                                          attacker_num,
                                                                                                          noise_string,
                                                                                                          lr_string,
                                                                                                          intrud, fgsm,
                                                                                                          actorAggOnly,
                                                                                                          rule),
            Steps)
    np.save(
        "data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s/normalAgents.npy" % (policy, task,
                                                                                                          env_name,
                                                                                                          numAgent,
                                                                                                          attacker_num,
                                                                                                          noise_string,
                                                                                                          lr_string,
                                                                                                          intrud,
                                                                                                          fgsm,
                                                                                                          actorAggOnly),
        normalAgents)


def run_multi_rules(actorAggOnly, policy="TD3", rules=("ResAgg",)):
    normalize = False
    softmax = True
    # normalize = True
    # softmax = False

    for rule in rules:
        # Set seeds
        # env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        Agents = [Agent(x, policy, env[x], lr[x], noise_amp[x], args, kwargs) for x in range(numAgent)]

        if args.FGSMattackAgent > 0:
            for x in range(args.FGSMattackAgent):
                Agents[x].fgsm = True

        print("lr: ", [Agents[x].lr for x in normalAgents])
        print("noise: ", [Agents[x].noise_amp for x in normalAgents])
        print("fgsm attack: ", [Agents[x].fgsm for x in normalAgents])

        global Reward, Steps
        Reward = np.zeros((numAgent, MAX_EPISODES))
        Steps = np.zeros((numAgent, MAX_EPISODES))
        # Reward_rules[rule] = Reward

        start_time = time.time()

        ep = 0
        # for ep in range(MAX_EPISODES):
        smallest_steps_by_far = 0

        while smallest_steps_by_far < args.max_timesteps:
            print()
            print('EPISODE: %d' % ep)

            Threads = [Thread(target=Agents[k].run_one_episode, args=(ep,)) for k in normalAgents]
            for t_k in Threads:
                t_k.start()
            for t_k in Threads:
                t_k.join()

            # if ep > 0:
            #     aggregate_neigh_models(agents, rule=rule)
            if ep >= 0 and rule != "no-coop":
                Threads_agg = [None] * len(normalAgents)
                if policy == "DDPG" or policy == "TD3":
                    models = {
                        "actor": [Agents[x].policy.actor for x in range(numAgent)],
                        "critic": [Agents[x].policy.critic for x in range(numAgent)],
                    }
                    target_models = {
                        "actor_target": [Agents[x].policy.actor_target for x in range(numAgent)],
                        "critic_target": [Agents[x].policy.critic_target for x in range(numAgent)],
                    }
                elif policy == "SAC":
                    models = {
                        "actor": [Agents[x].policy.policy for x in range(numAgent)],
                        "critic": [Agents[x].policy.qf1 for x in range(numAgent)],
                        "critic_2": [Agents[x].policy.qf2 for x in range(numAgent)],
                    }
                    target_models = {
                        "critic_target": [Agents[x].policy.qf1_target for x in range(numAgent)],
                        "critic_target_2": [Agents[x].policy.qf2_target for x in range(numAgent)],
                    }
                else:
                    raise NameError("Policy name is not defined!")
                Aggr_actor_state_dict, Aggr_critic_state_dict, Aggr_critic_state_dict_2 = {}, {}, {}
                # ActorDict, CriticDict = torch.rand(20, 30), torch.rand(20, 30)
                ActorDict, CriticDict, Critic2Dict, Actorlayers, CriticLayers = getModelDict(policy, models)
                Q_exchange = [Agents[x].ep_r for x in range(numAgent)]

                for k in normalAgents:
                    Threads_agg[k] = Thread(target=Agents[k].aggregate_models, args=(
                        rule, models, target_models, Q_exchange, deepcopy(ActorDict), deepcopy(CriticDict),
                        deepcopy(Critic2Dict), Actorlayers,
                        CriticLayers,
                        Neigh[k], Accumu_Q_actor, Accumu_Q_critic, Aggr_actor_state_dict,
                        Aggr_critic_state_dict, Aggr_critic_state_dict_2, args.filter, normalize, softmax, actorAggOnly,
                        alpha))
                    Threads_agg[k].start()

                for t_k in Threads_agg:
                    t_k.join()

                # TAU = 0.001
                TAU = 0.001
                # print(Aggr_actor_state_dict[0]['l1.weight'])
                for k in normalAgents:
                    if policy == "DDPG" or policy == "TD3":
                        Agents[k].policy.actor.load_state_dict(Aggr_actor_state_dict[k])
                        # print(torch.max(Aggr_actor_state_dict[k]['l1.weight'] - Aggr_actor_state_dict[0]['l1.weight']))
                        # target_Model[i].load_state_dict(Aggr_state_ct[i])
                        soft_update(Agents[k].policy.actor_target, Agents[k].policy.actor, TAU)
                        if not actorAggOnly:
                            Agents[k].policy.critic.load_state_dict(Aggr_critic_state_dict[k])
                            # target_Model[i].load_state_dict(Aggr_state_dict[i])
                            soft_update(Agents[k].policy.critic_target, Agents[k].policy.critic, TAU)
                    elif policy == "SAC":
                        Agents[k].policy.policy.load_state_dict(Aggr_actor_state_dict[k])
                        if not actorAggOnly:
                            Agents[k].policy.qf1.load_state_dict(Aggr_critic_state_dict[k])
                            soft_update(Agents[k].policy.qf1_target, Agents[k].policy.qf1, TAU)
                            Agents[k].policy.qf2.load_state_dict(Aggr_critic_state_dict_2[k])
                            soft_update(Agents[k].policy.qf2_target, Agents[k].policy.qf2, TAU)
                    else:
                        raise NameError("Policy name is not defined!")

                # print(Aggr_actor_state_dict[0]['l1.weight'])

            steps_by_far = [Agents[x].stepsbyfar for x in normalAgents]
            smallest_steps_by_far = np.min(steps_by_far)

            if smallest_steps_by_far > args.start_timesteps:

                Threads_eval = [Thread(target=Agents[k].eval_policy, args=(env_name[k], Reward, Steps, ep))
                                for k in
                                normalAgents]
                for t_k in Threads_eval:
                    t_k.start()
                for t_k in Threads_eval:
                    t_k.join()

                # # print(Accumu_reward)
                # print()
                print(
                    "Total time used by far for %d agents is: %.2f s" % (numAgent, time.time() - start_time))
                if ep > 0:
                    save_reward(Reward, Steps, rule, args.multi_task, args.env, normalAgents, noise_amp, lr,
                                args.filter, args.intruder, args.FGSMattackAgent, normalize, softmax, policy,
                                actorAggOnly)

                ep += 1


if __name__ == '__main__':

    # attacker_num_list = [0, 2, 4]
    # for attacker_num in attacker_num_list:
    attacker_num = 0
    random.seed(0)
    np.random.seed(0)

    # network
    numAgent = 20   #8  # 30

    # attacks
    attacker = random.sample(range(numAgent), k=attacker_num)
    normalAgents = [x for x in range(numAgent) if x not in attacker]
    print("attackers are: ", attacker)

    lower = 0
    upper = 6   #5   #3
    sensingRange = 1.5    #1.5  # 1  # 1.4  #2.8
    x = random_point_set(numAgent, lower=lower, upper=upper)

    Neigh = []
    neigh_sum = 0
    for k in range(numAgent):
        neighbor = findNeighbors(x, k, numAgent, sensingRange, maxNeighborSize=numAgent)
        Neigh.append(neighbor)
        if k in normalAgents:
            neigh_sum += len(neighbor)

    print(Neigh)
    average_neigh = neigh_sum / len(normalAgents)
    print("average neighbor number for normal agents is: %.2f" % average_neigh)
    degree = [len(Neigh[x]) for x in range(numAgent)]
    # print(degree)
    alpha = [(1 - sum([1 / max(degree[k], degree[l]) for l in Neigh[k] if l != k])) * degree[k] for k in
             range(numAgent)]
    # print(alpha)

    parser = argparse.ArgumentParser()
    parser.add_argument("--actorAggOnly", default=False)  # Aggregate only actor policy
    parser.add_argument("--FGSMattackAgent", default=0)  # Agent number under FGSM attack
    parser.add_argument("--multi_task", default=False)  # multi-task or not
    parser.add_argument("--noise_hetero", default=False)  # heterogeneous or homogeneous agentsc
    parser.add_argument("--intruder", default=0)  # heterogeneous or homogeneous agents
    parser.add_argument("--lr_hetero", default=False)  # heterogeneous or homogeneous agents
    parser.add_argument("--filter", default=False)  # cooperation with only those having a larger Q
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or SAC)
    parser.add_argument("--env", default="Walker2d-v2")  # OpenAI gym environment name  "Walker2d-v2" # HalfCheetah-v2 InvertedDoublePendulum-v2 Hopper-v3 Humanoid-v2""InvertedDoublePendulum-v2" "Ant-v2" "Reacher-v2"
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=125000, type=int)  # Max time steps to run environment  225000: this is the MAXIMUM NO. OF EPISODES
    parser.add_argument("--max_episodesteps", default=1e3, type=int)  # Max time steps in an episode
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model",
                        default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if args.noise_hetero:
        noise_amp = [random.randint(0, 10) * 1e-2 for _ in range(numAgent)]
    else:
        # noise_amp = [1e-2] * numAgent
        noise_amp = [0] * numAgent

    if args.multi_task:
        # env_name = ["Walker2d-v2", "HalfCheetah-v2", ] * 4
        env_name = ["Walker2d-v2", "HalfCheetah-v2", ] * 10
        # env_name = ["Humanoid-v2", "HumanoidStandup-v2"] * 4

        env = [gym.make(env_name[i]) for i in range(numAgent)]
        state_dim = env[0].observation_space.shape[0]
        action_dim = env[0].action_space.shape[0]
        max_action = float(env[0].action_space.high[0])
    else:
        env_name = [args.env] * numAgent
        env = [gym.make(env_name[i]) for i in range(numAgent)]
        state_dim = env[0].observation_space.shape[0]
        action_dim = env[0].action_space.shape[0]
        max_action = float(env[0].action_space.high[0])

    if args.lr_hetero:
        # lr = [random.randint(1, 9) * 1e-3 for _ in range(numAgent)]
        # lr = [3e-3] * 4 + [3e-2] * (numAgent - 4)
        lr = [1e-3] * 4 + [1e-4] * (numAgent - 4)
    else:
        # lr = [3e-3] * numAgent
        lr = [1e-3] * numAgent
        # lr = [3e-4] * numAgent

    if args.intruder > 0:
        for x in range(args.intruder):
            noise_amp[x] = 1

    MAX_EPISODES = 10000

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
    }

    if args.policy in ["DDPG", "TD3", "SAC"]:
        kwargs["max_action"] = max_action
        kwargs["tau"] = args.tau

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
    #     policy = TD3.TD3(**kwargs)
    # elif args.policy == "OurDDPG":
    #     policy = OurDDPG.DDPG(**kwargs)
    # elif args.policy == "DDPG":
    #     policy = DDPG.DDPG(**kwargs)

    # if args.load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    Accumu_Q_critic = np.zeros((numAgent, numAgent))
    Accumu_Q_actor = np.zeros((numAgent, numAgent))

    run_multi_rules(args.actorAggOnly, policy=args.policy, rules=("no-coop",  "median", "ResAgg", "average"))  #"no-coop",  "median", "ResAgg", "average"
    # run_multi_rules(args.actorAggOnly, policy=args.policy, rules=("average"))

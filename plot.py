import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from copy import deepcopy

# plt.style.use("fivethirtyeight")
# plt.style.use("fast")
plt.style.use("bmh")
# plt.style.use("seaborn-white")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 155
plt.rcParams['axes.facecolor'] = 'white'


def plot_subfigures(path0):
    # print(path0)
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # fig = plt.figure(figsize=(10, 3.5))
    fig = plt.figure(figsize=(5,4))
    # for i in range(3):
    try:

        attacker_num = attacker_num_list[0]
        path = path0 % (policy, multi_task,
                        env, numAgent, attacker_num, "_".join([str(x) for x in noise_amp]),
                        "_".join([str(x) for x in lr]), intruder, fgsm, actorAggOnly)
        print('path:', path)
        normalAgents = np.load("%s/normalAgents.npy" % path)

        mean_reward, max_reward, min_reward, steps = {}, {}, {}, {}
        mean_reward_copy, max_reward_copy, min_reward_copy = {}, {}, {}
        # for key in ["no-coop", "average", "median", "ResAgg_filter", "ResAgg_Q_filter", "ResAgg_no_filter",
        #             "ResAgg_filter_softmax", "ResAgg_Q_no_filter_softmax",
        #             "ResAgg_Q_no_filter", "ResAgg_no_filter_softmax",
                    # "ResAgg_no_filter_normalize_softmax", "ResAgg_no_filter_normalize"]:  # "ResAgg_noattack"
        for key in ["no-coop", "average", "median", "ResAgg_filter", "ResAgg_Q_filter", "ResAgg_no_filter",
                    "ResAgg_filter_softmax", "ResAgg_Q_no_filter_softmax",
                    "ResAgg_Q_no_filter", "ResAgg_no_filter_softmax",
                    "ResAgg_no_filter_normalize_softmax", "ResAgg_no_filter_normalize"]:  # "ResAgg_noattack"
            try:
                interval = 1
                span = 5
                steps[key] = np.load("%s/steps_%s.npy" % (path, key))[0]
                print('steps size:', np.shape(steps[key]))
                steps[key] -= 25000    # 25000
                length = len(steps[key])
                # length = 100
                steps[key] = [int(steps[key][x]) for x in range(length) if
                              x > 0 and steps[key][x] > 0 and x % interval == 0]
                Reward = np.load("%s/reward_%s.npy" % (path, key))
                # mean_reward[key] = np.mean(np.transpose(Reward[normalAgents]), 1)
                # mean_reward[key] = [mean_reward[key][x] for x in range(len(steps[key]))]
                # max_reward[key] = np.max(np.transpose(Reward[normalAgents]), 1)
                # max_reward[key] = [max_reward[key][x] for x in range(len(steps[key]))]
                # min_reward[key] = np.min(np.transpose(Reward[normalAgents]), 1)
                # min_reward[key] = [min_reward[key][x] for x in range(len(steps[key]))]

                # mean_reward[key] = np.mean(np.transpose(Reward[selectedAgent]), 1)
                # mean_reward[key] = [mean_reward[key][x] for x in range(0, len(steps[key]) * interval, interval)]
                # max_reward[key] = np.max(np.transpose(Reward[selectedAgent]), 1)
                # max_reward[key] = [max_reward[key][x] for x in range(0, len(steps[key]) * interval, interval)]
                # min_reward[key] = np.min(np.transpose(Reward[selectedAgent]), 1)
                # min_reward[key] = [min_reward[key][x] for x in range(0, len(steps[key]) * interval, interval)]

                mean_reward[key] = np.mean(np.transpose(Reward[selectedAgent]), 1)
                max_reward[key] = np.max(np.transpose(Reward[selectedAgent]), 1)
                min_reward[key] = np.min(np.transpose(Reward[selectedAgent]), 1)
                print('reward size', np.shape(Reward), 'mean reward share:', np.shape(mean_reward[key]))
                mean_reward_copy[key] = deepcopy(mean_reward[key][:len(steps[key])])
                max_reward_copy[key] = deepcopy(max_reward[key][:len(steps[key])])
                min_reward_copy[key] = deepcopy(min_reward[key][:len(steps[key])])
                print(length, 'mean reward copy shape:', np.shape(mean_reward_copy[key]))

                for x in range(0, len(steps[key]) * interval, interval):
                    if x < span:
                        mean_reward_copy[key][x] = np.mean(mean_reward[key][:x + span])
                    elif x >= span and x < len(steps[key]) - span:
                        mean_reward_copy[key][x] = np.mean(mean_reward[key][x - span:x + span])
                    else:
                        mean_reward_copy[key][x] = np.mean(mean_reward[key][x - span:len(steps[key])])

                for x in range(0, len(steps[key]) * interval, interval):
                    if x < span:
                        max_reward_copy[key][x] = np.mean(max_reward[key][:x + span])
                    elif x >= span and x < len(steps[key]) - span:
                        max_reward_copy[key][x] = np.mean(max_reward[key][x - span:x + span])
                    else:
                        max_reward_copy[key][x] = np.mean(max_reward[key][x - span:len(steps[key])])

                for x in range(0, len(steps[key]) * interval, interval):
                    if x < span:
                        min_reward_copy[key][x] = np.mean(min_reward[key][:x + span])
                    elif x >= span and x < len(steps[key]) - span:
                        min_reward_copy[key][x] = np.mean(min_reward[key][x - span:x + span])
                    else:
                        min_reward_copy[key][x] = np.mean(min_reward[key][x - span:len(steps[key])])

                # mean_reward_copy[key] = [
                #     np.mean(mean_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                #     mean_reward[key][x] for x in range(0, len(steps[key]) * interval, interval)]
                # max_reward_copy[key] = [
                #     np.mean(max_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                #     max_reward[key][x]
                #     for x in range(0, len(steps[key]) * interval, interval)]
                # min_reward_copy[key] = [
                #     np.mean(min_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                #     min_reward[key][x]
                #     for x in range(0, len(steps[key]) * interval, interval)]
            except:
                pass
        # line_style = {"no-coop": "-", "average": "-", "median": "-"}
        # ax = plt.subplot(1, 3, i + 1)
        # for key in mean_reward.keys():
        # for key in ["no-coop", "average", "median", "reward_max", "reward_softmax"]:
        line_style = {"no-coop": "-", "ResAgg_filter": "-", "ResAgg_Q_filter": "-", "ResAgg_no_filter": "-",
                      "ResAgg_Q_no_filter": "-", "average": "-", "median": "--", "ResAgg_noattack": "-",
                      "ResAgg_no_filter_softmax": "-", "ResAgg_no_filter_normalize_softmax": "-",
                      "ResAgg_filter_softmax": "-", "ResAgg_Q_no_filter_softmax": "-",
                      "ResAgg_no_filter_normalize": "-"}
        for key in ["no-coop", "ResAgg_no_filter_softmax", "average", "median", "ResAgg_no_filter_normalize"
                    ]:  # "ResAgg_filter" "ResAgg_no_filter", "ResAgg_no_filter_normalize_softmax",
        # for key in ["no-coop", "average", "median"]:
            try:
                print('steps key size', np.shape(steps[key]))
                plt.plot(steps[key], mean_reward_copy[key], linestyle=line_style[key])
                plt.fill_between(steps[key], min_reward_copy[key], max_reward_copy[key], alpha=0.3)
                plt.tick_params(labelsize=20)
                # plt.plot(mean_reward[key], label=key, linestyle=line_style[key])
                # plt.fill_between(range(len(min_reward[key])), min_reward[key], max_reward[key], alpha=0.3)
            except:
                pass
        # plt.xlabel(r'Time steps', fontstyle="italic")
        # plt.ylabel(r'Average Return', fontstyle="italic")
        plt.xlabel(r'Time steps', fontsize=25)
        plt.ylabel(r'Average Return', fontsize=25)
        # plt.legend("no-coop", "ResAgg_no_filter_softmax", "average", "median")
        # plt.xlim([0, 75000])
        # ax.set_title(title[i], y=-0.4)
        plt.tight_layout()
        # plt.savefig("fgfhg.jpg")

    except:
        pass

    # plt.ylim([-250,4500])
    plt.legend(ncol=5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.legend("no-coop", "ResAgg_no_filter_softmax", "average", "median")
    plt.show()
    # fig1.savefig('fig/MSD_mobile_attack%d.png' % attackerNum)


if __name__ == '__main__':
    env = 'Reacher-v2'  # "Ant-v2" #'HalfCheetah-v2'  ##  'InvertedDoublePendulum-v2' #'Reacher-v2' #  'Hopper-v3' #   #  'HalfCheetah-v2' #"InvertedDoublePendulum-v2" #'Humanoid-v2'  #  # 'Hopper-v3'
    policy = 'DDPG'  # "TD3" # "SAC"  #DDPG
    numAgent = 20   #8
    multi_task = "" #"multi_task_" #""  #   #
    intruder = ""  # "_4intruder"  #
    fgsm = "_19fgsm"  #""  #  # _7fgsm  _19fgsm
    # fgsm=""
    actorAggOnly = ""  # "_actorAggOnly" #
    attacker_num_list = [0]  # [0, 10, 29]
    # title = ["(a) no attack", "(b) %d Byzantine agents" % attacker_num_list[1],
    #          "(c) %d Byzantine agents" % attacker_num_list[2]]
    # ep = 20000 #202
    max_steps = 125000  # 45000
    # noise_amp = [0.08, 0.09, 0.02, 0.04, 0.01, 0.01, 0.1, 0.05]
    # noise_amp = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    noise_amp = [0] * numAgent
    # noise_amp = [1,1,0,0,0,0,0,0]
    # noise_amp = [1,1,1,1,1,1,1,0]
    # noise_amp = [1, 1, 1, 1, 0, 0, 0, 0]

    lr = [0.001] * numAgent
    # selectedAgent = list(range(numAgent))
    selectedAgent = list(range(19,numAgent))
    # selectedAgent = [19]
    print(selectedAgent)
    # selectedAgent = [7]
    # selectedAgent = [1, 3, 5, 7]
    # selectedAgent = [0, 2, 4, 6]
    # selectedAgent = [4, 5, 6, 7]
    # selectedAgent = [2,3,4,5,6,7]
    # homogeneous agents
    # path = "/home/jiani/Desktop/resilientActorCritic/mujoco/data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s"
    path = "/home/bhowmic/Downloads/Code downloads/resilientActorCritic-main_new/resilientActorCritic-main/mujoco/data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s"
    # path = "/home/bhowmic/Downloads/Code downloads/resilientActorCritic-main_new/resilientActorCritic-main/mujoco/data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s"
    # path = "/home/bhowmic/Downloads/Code downloads/resilientActorCritic-main_new/resilientActorCritic-main/mujoco/data_backup_large_conn4.8_Dec21/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s"
    # print(path)
    plot_subfigures(path)

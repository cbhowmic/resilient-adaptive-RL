import numpy as np
from matplotlib import pyplot as plt
import matplotlib

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
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # fig = plt.figure(figsize=(10, 3.5))
    fig = plt.figure(figsize=(4, 3.5))
    # for i in range(3):
    try:
        attacker_num = attacker_num_list[0]
        seprate_index = 0
        for row in range(len(row_list)):
            print("{:<15} ".format(row_list[row]), end='')
            if row <= 3:
                multi_task = ""
                env = env_list[row]
                fgsm = ""
            elif row_list[row] == "Multitask":
                multi_task = "multi_task_"
                env = env_list[0]
                fgsm = ""
            elif 5 <=row <= 7:
                multi_task = ""
                env = env_list[0]
                if row == 5:
                    fgsm = "_2fgsm"
                    seprate_index = 2
                elif row == 6:
                    fgsm = "_4fgsm"
                    seprate_index = 4
                else:
                    fgsm = "_7fgsm"
                    seprate_index = 7
            elif 8 <=row <= 10: #Ant
                multi_task = ""
                env = env_list[2]
                if row == 8:
                    fgsm = "_2fgsm"
                    seprate_index = 2
                elif row == 9:
                    fgsm = "_4fgsm"
                    seprate_index = 4
                else:
                    fgsm = "_7fgsm"
                    seprate_index = 7
            elif 11 <= row <= 13:  #Walk
                multi_task = ""
                env = env_list[1]
                if row == 11:
                    fgsm = "_2fgsm"
                    seprate_index = 2
                elif row == 12:
                    fgsm = "_4fgsm"
                    seprate_index = 4
                else:
                    fgsm = "_7fgsm"
                    seprate_index = 7
            elif 11 <= row <= 13:  # Reacher
                multi_task = ""
                env = env_list[3]
                if row == 11:
                    fgsm = "_2fgsm"
                    seprate_index = 2
                elif row == 12:
                    fgsm = "_4fgsm"
                    seprate_index = 4
                else:
                    fgsm = "_7fgsm"
                    seprate_index = 7

            for policy_i in range(len(policy_list)):

                path = path0 % (policy_list[policy_i], multi_task,
                                env, numAgent, attacker_num, "_".join([str(x) for x in noise_amp]), "_".join([str(x) for x in lr]), intruder, fgsm, actorAggOnly)

                mean_reward, max_reward, min_reward, steps = {}, {}, {}, {}

                for key in ["no-coop", "average", "median", "ResAgg_no_filter_softmax"]:  # "ResAgg_noattack"
                    try:
                        interval = 1
                        span = 5
                        steps[key] = np.load("%s/steps_%s.npy" % (path, key))[0]
                        steps[key] -= 25000
                        length = len(steps[key])
                        # length = 50
                        steps[key] = [int(steps[key][x]) for x in range(length) if
                                      x > 0 and steps[key][x] > 0 and x % interval == 0]
                        Reward = np.load("%s/reward_%s.npy" % (path, key))
                        avg, std = result(Reward, seprate_index)
                        tmp = str("%.2f" % avg) +"~$\pm$"+ str("%.2f" % std)
                        print("%15s" % tmp, end='')

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
                        mean_reward[key] = [
                            np.mean(mean_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                            mean_reward[key][
                                x] for x in range(0, len(steps[key]) * interval, interval)]
                        max_reward[key] = np.max(np.transpose(Reward[selectedAgent]), 1)
                        max_reward[key] = [
                            np.mean(max_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                            max_reward[key][x]
                            for x in range(0, len(steps[key]) * interval, interval)]
                        min_reward[key] = np.min(np.transpose(Reward[selectedAgent]), 1)
                        min_reward[key] = [
                            np.mean(min_reward[key][x - span:x + span]) if x >= span and x < len(steps[key]) - span else
                            min_reward[key][x]
                            for x in range(0, len(steps[key]) * interval, interval)]
                    except:
                        print("None ", end='')
                print("**", end='')
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
                    try:
                        plt.plot(steps[key], mean_reward[key], label=key, linestyle=line_style[key])
                        plt.fill_between(steps[key], min_reward[key], max_reward[key], alpha=0.3)
                        # plt.plot(mean_reward[key], label=key, linestyle=line_style[key])
                        # plt.fill_between(range(len(min_reward[key])), min_reward[key], max_reward[key], alpha=0.3)
                    except:
                        pass
                # plt.xlabel(r'Time steps', fontstyle="italic")
                # plt.ylabel(r'Average Return', fontstyle="italic")
                ##plt.xlabel(r'Time steps')
                ##plt.ylabel(r'Average Return')
                # plt.xlim([0, max_steps])
                # ax.set_title(title[i], y=-0.4)
                ##plt.tight_layout()

            print("\n")
    except:
        pass

            # plt.ylim([-250,4500])
            # plt.legend(ncol=5)
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ##plt.legend()
            ##plt.show()
            # fig1.savefig('fig/MSD_mobile_attack%d.png' % attackerNum)



def result(Reward, seprate_index):
    # non_zero_idex = 100
    non_zero_idex = 1000
    Reward = Reward[seprate_index:, :]
    for i in range(len(Reward)):
        non_zero_idex_i = np.where(Reward[i] == 0)[0]
        non_zero_idex_i = non_zero_idex_i[0]
        non_zero_idex = min(non_zero_idex_i, non_zero_idex)

    Reward = Reward[:, 0:non_zero_idex]
    max_arr = np.max(Reward, 1)
    avg = np.average(max_arr)
    std = np.std(max_arr)
    return avg, std


if __name__ == '__main__':
    row_list = np.array(['HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Reacher-v2', 'Multitask', "HalfCheetah-2-fgsm", "HalfCheetah-4-fgsm", "HalfCheetah-7-fgsm",
                         "Ant-2-fgsm", "Ant-4-fgsm","Ant-7-fgsm",
                         "Walker2d-2-fgsm", "Walker2d-4-fgsm", "Walker2d-7-fgsm",
                         "Reacher-2-fgsm", "Reacher-4-fgsm", "Reacher-7-fgsm"])
    env_list = np.array(['HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Reacher-v2'])
    policy_list = np.array(['DDPG', 'SAC', 'TD3'])
    print("{:^15}{:^60}{:^60}{:^60}".format("", "DDPG", "SAC","TD3"))
    print("{:^15}{:^15}{:^15}{:^15}{:^15}*{:^15}{:^15}{:^15}{:^15}*{:^15}{:^15}{:^15}{:^15}*".format("", "no-coop", "average","median", "Ours",
                                                                                            "no-coop","average", "median", "Ours",
                                                                                            "no-coop", "average","median", "Ours"))

    # env = "HalfCheetah-v2" #'Reacher-v2'  # 'HalfCheetah-v2'  # 'InvertedDoublePendulum-v2' #'Reacher-v2' #  'Hopper-v3' #   #  'HalfCheetah-v2' #"InvertedDoublePendulum-v2" #'Humanoid-v2'  #  # 'Hopper-v3'
    # policy = 'DDPG'  # "TD3" # "SAC"  #
    numAgent = 8
    multi_task = ""  #"multi_task_"  #
    intruder = ""  # "_4intruder"  #
    fgsm = ""  # "_2fgsm"  # #
    actorAggOnly = ""  # "_actorAggOnly" #
    attacker_num_list = [0, 2, 4]  # [0, 10, 29]
    title = ["(a) no attack", "(b) %d Byzantine agents" % attacker_num_list[1],
             "(c) %d Byzantine agents" % attacker_num_list[2]]
    # ep = 20000 #202
    max_steps = 225000  # 45000
    # noise_amp = [0.08, 0.09, 0.02, 0.04, 0.01, 0.01, 0.1, 0.05]
    # noise_amp = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    noise_amp = [0] * numAgent
    # noise_amp = [1,1,0,0,0,0,0,0]
    # noise_amp = [1,1,1,1,1,1,1,0]
    # noise_amp = [1, 1, 1, 1, 0, 0, 0, 0]

    lr = [0.001] * numAgent
    selectedAgent = list(range(numAgent))
    # selectedAgent = [7]
    # selectedAgent = [1, 3, 5, 7]
    # selectedAgent = [0, 2, 4, 6]
    # selectedAgent = [4, 5, 6, 7]
    # selectedAgent = [2,3,4,5,6,7]
    # homogeneous agents
    path = "/home/jiani/Desktop/resilientActorCritic/mujoco/data/policy_%s/reward_dict_%s%s_%d_agents_%d_attackers_noise_%s_lr_%s%s%s%s"

    Reward = plot_subfigures(path)
from __future__ import division

import random
from algo.utils import ReplayBuffer
from algo import DDPG, TD3, PPO, SAC
from aggregateMethods import *
import gym
import gc
from FGSM_attack import fgsm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, agent_id, policy_name, env, lr, noise_amp, args, kwargs):
        self.agent_id = agent_id
        self.policyName = policy_name
        self.env = deepcopy(env)
        self.env.seed(random.randint(0, 100))
        self.action_dim = kwargs["action_dim"]
        self.state_dim = kwargs["state_dim"]
        self.max_action = kwargs['max_action']
        self.ram = ReplayBuffer(self.state_dim, self.action_dim)
        self.args = args
        self.lr = lr
        self.noise_amp = noise_amp
        self.fgsm = False
        if self.policyName == "TD3" or self.policyName == "PPO":
            kwargs["lr"] = self.lr
        if self.policyName == "TD3":
            self.policy = TD3.TD3(**kwargs)
        elif self.policyName == "DDPG":
            self.policy = DDPG.DDPG(**kwargs)
        elif self.policyName == "PPO":
            self.policy = PPO.PPO(**kwargs)
        elif self.policyName == "SAC":
            self.policy = SAC.SAC(**kwargs)
        else:
            raise NameError("Policy name is not defined!")
        self.stepsbyfar = 0
        self.ep_r = 0

    def aggregate_models(self, rule, models, target_models, Q_exchange, ActorDict, CriticDict, Critic2Dict, Actorlayers,
                         CriticLayers, neighbors,
                         Accumu_loss_actor, Accumu_loss_critic, Aggr_actor_state_dict, Aggr_critic_state_dict,
                         Aggr_critic_state_dict_2, filter,
                         normalize, softmax, actorAggOnly, alpha):

        if self.policyName == "DDPG" or self.policyName == "TD3":
            Model_actor, Model_critic, Model_critic_2 = models["actor"], models["critic"], None
            Model_target_actor, Model_target_critic, Model_target_critic_2 = target_models["actor_target"], \
                                                                             target_models["critic_target"], None
        elif self.policyName == "SAC":
            Model_actor, Model_critic, Model_critic_2 = models["actor"], models["critic"], models["critic_2"]
            Model_target_actor, Model_target_critic, Model_target_critic_2 = None, \
                                                                             target_models["critic_target"], \
                                                                             target_models["critic_target_2"]
        else:
            raise NameError("Policy name is not defined!")

        for Model, target_Model, ModelDict, keys in zip([Model_actor, Model_critic],
                                                        [Model_target_actor, Model_target_critic],
                                                        [ActorDict, CriticDict], [Actorlayers, CriticLayers]):
            if rule == "average":
                if Model == Model_actor:
                    Aggr_actor_state_dict[self.agent_id] = average_rule(keys, ModelDict, neighbors)
                elif Model == Model_critic and not actorAggOnly:
                    # Aggr_critic_state_dict[self.agent_id] = Model[self.agent_id].state_dict()
                    Aggr_critic_state_dict[self.agent_id] = average_rule(keys, ModelDict, neighbors)
                    if self.policyName == "SAC":
                        Aggr_critic_state_dict_2[self.agent_id] = average_rule(keys, Critic2Dict, neighbors)
            elif rule == "median":
                if Model == Model_actor:
                    Aggr_actor_state_dict[self.agent_id] = median_rule(keys, ModelDict, neighbors)
                elif Model == Model_critic and not actorAggOnly:
                    # Aggr_critic_state_dict[self.agent_id] = Model[self.agent_id].state_dict()
                    Aggr_critic_state_dict[self.agent_id] = median_rule(keys, ModelDict, neighbors)
                    if self.policyName == "SAC":
                        Aggr_critic_state_dict_2[self.agent_id] = median_rule(keys, Critic2Dict, neighbors)
            elif rule == "ResAgg":
                if Model == Model_actor:
                    Aggr_actor_state_dict[self.agent_id] = actor_rule(self.agent_id, self.policyName, Model_actor,
                                                                      Model_critic, Model_critic_2, self.ram, keys,
                                                                      ActorDict,
                                                                      neighbors, alpha, Accumu_loss_actor, filter, normalize,
                                                                      softmax)
                    # Aggr_actor_state_dict[self.agent_id] = average_rule(keys, ModelDict, neighbors)
                elif Model == Model_critic and not actorAggOnly:
                    ## Aggr_critic_state_dict[self.agent_id] = Model[self.agent_id].state_dict()
                    if self.policyName == "SAC":
                        Aggr_critic_state_dict[self.agent_id], Aggr_critic_state_dict_2[self.agent_id] = critic_rule(
                            self.agent_id, self.policyName, Model_actor,
                            Model_critic, Model_critic_2, Model_target_critic, Model_target_critic_2, self.ram,
                            keys, CriticDict, Critic2Dict,
                            neighbors, alpha, Accumu_loss_critic, filter, softmax)
                    else:
                        Aggr_critic_state_dict[self.agent_id] = critic_rule(self.agent_id, self.policyName, Model_actor,
                                                                            Model_critic, Model_critic_2,
                                                                            Model_target_critic, Model_target_critic_2,
                                                                            self.ram,
                                                                            keys, CriticDict, Critic2Dict,
                                                                            neighbors, alpha, Accumu_loss_critic, filter, softmax)
                    # Aggr_critic_state_dict[self.agent_id] = average_rule(keys, ModelDict, neighbors)
                    # if self.policyName == "SAC":
                    #     Aggr_critic_state_dict_2[self.agent_id] = average_rule(keys, Critic2Dict, neighbors)
            else:
                raise Exception("Rule not defined!")

    def eval_policy(self, env_name, Reward, Steps, ep):
        eval_env = gym.make(env_name)
        eval_env.seed(self.args.seed)

        state, done = eval_env.reset(), False
        eval_reward = 0
        while not done:
            if self.policyName == "DDPG" or self.policyName == "TD3":
                action = self.policy.select_action(np.array(state))
            elif self.policyName == "PPO":
                mu, sigma = self.policy.pi(torch.from_numpy(state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
            elif self.policyName == "SAC":
                action, _, _ = self.policy.policy(torch.Tensor(state).to(device))
                action = action.detach().cpu().numpy()
            else:
                raise NameError("Policy name is not defined!")
            state, reward, done, _ = eval_env.step(action)
            eval_reward += reward

        print(f"Evaluation reward for agent {self.agent_id}: {eval_reward:.3f}")

        Reward[self.agent_id, ep] = eval_reward
        Steps[self.agent_id, ep] = self.stepsbyfar

        return eval_reward

    def run_one_episode(self, ep):
        # self.trainer.load_agent_model(self.agent_id, 100)
        # run(self.env, self.trainer, self.ram, self.agent_id, MAX_EPISODES, MAX_STEPS)
        # if self.agent_id == 0:
        #     print('EPISODE: %d' % ep)
        state, done = self.env.reset(), False
        self.ep_r = 0
        # while True:
        for t in range(int(self.args.max_episodesteps) + 1):  # time steps within an episode
            if self.fgsm:
                if self.policyName == "DDPG" or self.policyName == "TD3":
                    state = fgsm(self.policyName, self.policy.critic, None, self.policy.actor, state)
                elif self.policyName == "SAC":
                    state = fgsm(self.policyName, self.policy.qf1, self.policy.qf2, self.policy.policy, state)
                else:
                    raise NameError("Policy name is not defined!")
            else:
                state += np.random.random_sample((self.state_dim)) * self.noise_amp
            # Select action randomly or according to policy

            if self.policyName == "DDPG" or self.policyName == "TD3":
                if t + self.stepsbyfar < self.args.start_timesteps:
                    action = self.env.action_space.sample()
                else:
                    action = (
                            self.policy.select_action(np.array(state))
                            + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
                    ).clip(-self.max_action, self.max_action)
            elif self.policyName == "PPO":
                mu, sigma = self.policy.pi(torch.from_numpy(state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            elif self.policyName == "SAC":
                if t + self.stepsbyfar < self.args.start_timesteps:
                    action = self.env.action_space.sample()
                else:
                    _, action, _ = self.policy.policy(torch.Tensor(state).to(device))
                    action = action.detach().cpu().numpy()
            else:
                raise NameError("Policy name is not defined!")

            # Perform action
            next_state, reward, done, _ = self.env.step(action)
            done_bool = float(done) if t < self.env._max_episode_steps else 0

            # Store data in replay buffer
            if self.policyName == "DDPG" or self.policyName == "TD3" or self.policyName == "SAC":
                self.ram.add(state, action, next_state, reward, done_bool)
                # Train agent after collecting sufficient data
                if t + self.stepsbyfar >= self.args.start_timesteps:
                    self.policy.train(self.ram, self.args.batch_size)
            elif self.policyName == "PPO":
                # self.policy.put_data((state, action, reward / 10.0, next_state, log_prob.detach().cpu().numpy(), done_bool))
                self.policy.put_data((state, action, reward, next_state, log_prob.detach().cpu().numpy(), done_bool))
                self.policy.train()
            else:
                raise NameError("Policy name is not defined!")

            state = next_state
            self.ep_r += reward

            if done:
                # self.stepsbyfar += t + 1
                # print(
                #     f"Total T: {self.stepsbyfar} for agent {self.agent_id}, Episode Num: {ep} Reward: {self.ep_r:.3f}")
                # # Reset environment
                state, done = self.env.reset(), False
                # break

        self.stepsbyfar += t
        print(
            f"Total T: {self.stepsbyfar} for agent {self.agent_id}, Episode Num: {ep} Reward: {self.ep_r:.3f}")

        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)

import argparse
import os
import sys
import random
import numpy as np
import scipy
 
from params import Params
 
import pickle
import time
 
import statistics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from operator import add, sub
 
import pickle
 
 
class PPOStorage:
    def __init__(self, num_inputs, num_outputs, max_size=64000):
        self.states = torch.zeros(max_size, num_inputs).to(device)
        self.next_states = torch.zeros(max_size, num_inputs).to(device)
        self.actions = torch.zeros(max_size, num_outputs).to(device)
        self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
        self.log_probs = torch.zeros(max_size).to(device)
        self.rewards = torch.zeros(max_size).to(device)
        self.q_values = torch.zeros(max_size, 1).to(device)
        self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
        self.counter = 0
        self.sample_counter = 0
        self.max_samples = max_size
    def sample(self, batch_size):
        idx = torch.randint(self.counter, (batch_size,),device=device)
        return self.states[idx, :], self.actions[idx, :], self.next_states[idx, :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]
    def clear(self):
        self.counter = 0
    def push(self, states, actions, next_states, rewards, q_values, log_probs, size):
        self.states[self.counter:self.counter+size, :] = states.detach().clone()
        self.actions[self.counter:self.counter+size, :] = actions.detach().clone()
        self.next_states[self.counter:self.counter+size, :] = next_states.detach().clone()
        self.rewards[self.counter:self.counter+size] = rewards.detach().clone()
        self.q_values[self.counter:self.counter+size, :] = q_values.detach().clone()
        self.log_probs[self.counter:self.counter+size] =  log_probs.detach().clone()
        self.counter += size

    def critic_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter-batch_size:self.sample_counter, :], self.q_values[self.sample_counter-batch_size:self.sample_counter, :]
    
    def actor_sample(self, batch_size):
        if self.sample_counter == 0 or self.sample_counter == self.max_samples:
            self.permute()
        self.sample_counter %= self.max_samples
        self.sample_counter += batch_size
        return self.states[self.sample_counter-batch_size:self.sample_counter, :], self.actions[self.sample_counter-batch_size:self.sample_counter, :], self.q_values[self.sample_counter-batch_size:self.sample_counter, :], self.log_probs[self.sample_counter-batch_size:self.sample_counter]

    def permute(self):
        permuted_index = torch.randperm(self.max_samples)
        self.states[:, :] = self.states[permuted_index, :]
        self.actions[:, :] = self.actions[permuted_index, :]
        self.q_values[:, :] = self.q_values[permuted_index, :]
        self.log_probs[:] = self.log_probs[permuted_index]
 
class RL(object):
    def __init__(self, env, hidden_layer=[64, 64]):
        self.env = env
        self.num_inputs = env.observation_space.shape[0]
        self.num_outputs = env.action_space.shape[0]
        self.hidden_layer = hidden_layer
 
        self.params = Params()
        self.Net = ActorCriticNet
        self.model = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer)
        self.model.share_memory()
        self.test_mean = []
        self.test_std = []
 
        self.noisy_test_mean = []
        self.noisy_test_std = []

        # reward logging
        self.noisy_test_mean_by_name = {}
        self.noisy_test_std_by_name = {}

        # termination logging
        self.trigger_count = []

        self.fig = plt.figure(figsize=(16, 10))
        #self.fig2 = plt.figure()
        self.lr = self.params.lr
        plt.show(block=False)
 
        self.test_list = []
        self.noisy_test_list = []
 
        self.gpu_model = self.Net(self.num_inputs, self.num_outputs,self.hidden_layer)
        self.gpu_model.to(device)
        self.model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer).to(device)
 
        self.base_controller = None
        self.base_policy = None
 
        self.total_rewards = []
        # reward logging
        self.reward_info = []
        self.total_reward_info = {}


    def normalize_data(self, num_iter=1000, file='shared_obs_stats.pkl'):
        state = self.env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        for i in range(num_iter):
            print(i)
            self.shared_obs_stats.observes(state)
            state = self.shared_obs_stats.normalize(state)#.to(device)
            env_action = np.random.randn(self.num_outputs)
            state, reward, done, _ = self.env.step(env_action*0)
    
            if done:
                state = self.env.reset()
 
            state = Variable(torch.Tensor(state).unsqueeze(0))
 
        with open(file, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)
 
    def run_test(self, num_test=1):
        state = self.env.reset()
        ave_test_reward = 0
    
        total_rewards = []
        if self.num_envs > 1:
            test_index = 1
        else:
            test_index = 0
    
        for i in range(num_test):
            total_reward = 0
            while True:
                state = self.shared_obs_stats.normalize(state)
                mu = self.gpu_model.sample_best_actions(state)
                state, reward, done, _ = self.env.step(mu)
                total_reward += reward[test_index].item()
    
                if done[test_index]:
                    state = self.env.reset()
                    ave_test_reward += total_reward / num_test
                    total_rewards.append(total_reward)
                    break
        reward_mean = statistics.mean(total_rewards)
        reward_std = statistics.stdev(total_rewards)
        self.test_mean.append(reward_mean)
        self.test_std.append(reward_std)
        self.test_list.append((reward_mean, reward_std))
        #print(self.model.state_dict())
    
    def run_test_with_noise(self, num_test=10):
        reward_mean = statistics.mean(self.total_rewards)
        reward_std = statistics.stdev(self.total_rewards)
        #print(reward_mean, reward_std, self.total_rewards)
        self.noisy_test_mean.append(reward_mean)
        self.noisy_test_std.append(reward_std)
        self.noisy_test_list.append((reward_mean, reward_std))

        for key in self.total_reward_info.keys():
            if not(key in self.noisy_test_mean_by_name): self.noisy_test_mean_by_name[key]  = []
            if not(key in self.noisy_test_std_by_name): self.noisy_test_std_by_name[key]  = []
            self.noisy_test_mean_by_name[key].append(statistics.mean(self.total_reward_info[key]))
            self.noisy_test_std_by_name[key].append(statistics.stdev(self.total_reward_info[key]))


        # trigger logging
        total_triggers = np.sum(self.env._done_count)
        ratio = [trigger / total_triggers for trigger in self.env._done_count]
        self.trigger_count.append(ratio)
        self.env._done_count = np.zeros(self.env._done_count.shape[0])
    
        print("reward mean,", reward_mean)
        print("reward std,", reward_std)
    
    def save_reward_stats(self, stats_name):
        with open( stats_name, 'wb') as f:
            np.save(f, np.array(self.noisy_test_mean))
            np.save(f, np.array(self.noisy_test_std))
    


    def plot_statistics(self):
        plt.clf()

        index = []
        noisy_low = []
        noisy_high = []
        noisy_low_by_name = {}
        noisy_high_by_name = {}
        for i in range(len(self.noisy_test_mean)):
            noisy_low.append(self.noisy_test_mean[i]-self.noisy_test_std[i])
            noisy_high.append(self.noisy_test_mean[i]+self.noisy_test_std[i])

            for key in self.total_reward_info.keys():
                if not(key in noisy_low_by_name): noisy_low_by_name[key] = []
                if not(key in noisy_high_by_name): noisy_high_by_name[key] = []
                noisy_low_by_name[key].append(self.noisy_test_mean_by_name[key][i] - self.noisy_test_std_by_name[key][i])
                noisy_high_by_name[key].append(self.noisy_test_mean_by_name[key][i] + self.noisy_test_std_by_name[key][i])

            index.append(i)
        
        ax1 = self.fig.add_subplot(241)
        ax1.set_title("orientation reward")
        ax1.plot(self.noisy_test_mean_by_name["orientation"], 'g')
        ax1.fill_between(index, noisy_low_by_name["orientation"], noisy_high_by_name["orientation"], color='r')

        ax2 = self.fig.add_subplot(242)
        ax2.set_title("velocity reward")
        ax2.plot(self.noisy_test_mean_by_name["velocity"], 'g')
        ax2.fill_between(index, noisy_low_by_name["velocity"], noisy_high_by_name["velocity"], color='r')

        ax3 = self.fig.add_subplot(243)
        ax3.set_title("end-effector reward")
        ax3.plot(self.noisy_test_mean_by_name["end effector"], 'g')
        ax3.fill_between(index, noisy_low_by_name["end effector"], noisy_high_by_name["end effector"], color='r')

        ax4 = self.fig.add_subplot(244)
        ax4.set_title("center-of-mass reward")
        ax4.plot(self.noisy_test_mean_by_name["com"], 'g')
        ax4.fill_between(index, noisy_low_by_name["com"], noisy_high_by_name["com"], color='r')

        ax5 = self.fig.add_subplot(245)
        ax5.set_title("ball contact reward")
        ax5.plot(self.noisy_test_mean_by_name["contact"], 'g')
        ax5.fill_between(index, noisy_low_by_name["contact"], noisy_high_by_name["contact"], color='r')

        ax6 = self.fig.add_subplot(246)
        ax6.set_title("ball distance reward")
        ax6.plot(self.noisy_test_mean_by_name["ball distance"], 'g')
        ax6.fill_between(index, noisy_low_by_name["ball distance"], noisy_high_by_name["ball distance"], color='r')

        ax7 = self.fig.add_subplot(247)
        ax7.set_title("total reward")
        ax7.plot(self.noisy_test_mean, 'g')
        ax7.fill_between(index, noisy_low, noisy_high, color='r')
        
        ax8 = self.fig.add_subplot(248)
        ax8.set_title("triggered condition")
        ax8.plot([ratio[0] for ratio in self.trigger_count], 'r')
        ax8.plot([ratio[1] for ratio in self.trigger_count], 'g')
        ax8.plot([ratio[2] for ratio in self.trigger_count], 'b')
        ax8.plot([ratio[3] for ratio in self.trigger_count], 'black')

        self.fig.canvas.draw()

        np.savetxt(self.model_name + "orientation_mean.txt", np.array(self.noisy_test_mean_by_name["orientation"]))
        np.savetxt(self.model_name + "orientation_low.txt", np.array(self.noisy_test_std_by_name["orientation"]))

        np.savetxt(self.model_name + "velocity_mean.txt", np.array(self.noisy_test_mean_by_name["velocity"]))
        np.savetxt(self.model_name + "velocity_low.txt",        np.array(self.noisy_test_std_by_name["velocity"]))

        np.savetxt(self.model_name + "endeffector_mean.txt", np.array(self.noisy_test_mean_by_name["end effector"]))
        np.savetxt(self.model_name + "endeffector_low.txt",        np.array(self.noisy_test_std_by_name["end effector"]))

        np.savetxt(self.model_name + "com_mean.txt", np.array(self.noisy_test_mean_by_name["com"]))
        np.savetxt(self.model_name + "com_low.txt",        np.array(self.noisy_test_std_by_name["com"]))

        np.savetxt(self.model_name + "contact_mean.txt", np.array(self.noisy_test_mean_by_name["contact"]))
        np.savetxt(self.model_name + "contact_low.txt",        np.array(self.noisy_test_std_by_name["contact"]))

        np.savetxt(self.model_name + "balldistance_mean.txt", np.array(self.noisy_test_mean_by_name["ball distance"]))
        np.savetxt(self.model_name + "balldistance_low.txt",        np.array(self.noisy_test_mean_by_name["ball distance"]))

        np.savetxt(self.model_name + "total_mean.txt", np.array(self.noisy_test_mean))
        np.savetxt(self.model_name + "total_low.txt",        np.array(self.noisy_test_std))

        np.savetxt(self.model_name + "triggercount.txt", np.array(self.trigger_count))
    
    def collect_samples_vec(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):
        #    print("COLLECT SAMPLES..")
        start_state = self.env.observe()
        samples = 0
        done = False
        states = []
        next_states = []
        actions = []
        rewards = []
        values = []
        q_values = []
        real_rewards = []
        log_probs = []
        dones = []
        noise = self.base_noise * self.explore_noise.value
        self.gpu_model.set_noise(noise)
 
        state = start_state
        total_reward1 = 0
        total_reward2 = 0
        calculate_done1 = False
        calculate_done2 = False
        self.total_rewards = []
        start = time.time()
        while samples < num_samples:
            with torch.no_grad():
                action, mean_action = self.gpu_model.sample_actions(state)
                log_prob = self.gpu_model.calculate_prob(state, action, mean_action)
    
            states.append(state.clone())
            actions.append(action.clone())
            log_probs.append(log_prob.clone())
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward.clone())
    
            dones.append(done.clone())
    
            next_states.append(state.clone())
    
            samples += 1
    
            self.env.reset_time_limit()
            #    print(f"{samples}/{num_samples} samples collected")
        print("sim time", time.time() - start)
        start = time.time()
        counter = num_samples - 1
        R = self.gpu_model.get_value(state)
        while counter >= 0:
            R = R * (1 - dones[counter].unsqueeze(-1))
            R = 0.995 * R + rewards[counter].unsqueeze(-1)
            q_values.insert(0, R)
            counter -= 1
            #print(len(q_values))
        for i in range(num_samples):
            self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)
        self.total_rewards = self.env.total_rewards.cpu().numpy().tolist()
        print("processing time", time.time() - start)
        # reward logging
        self.reward_info = self.env.reward_info
        for key in self.reward_info[0].keys():
            self.total_reward_info[key] = []
            for i in range(self.num_envs):
                self.total_reward_info[key].append(self.reward_info[i][key])

    
    def update_critic(self, batch_size, num_epoch):
        self.gpu_model.train()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=10*self.lr)

        storage = self.storage
        gpu_model = self.gpu_model

        for k in range(num_epoch):
            batch_states, batch_q_values = storage.critic_sample(batch_size)
            batch_q_values = batch_q_values
            v_pred = gpu_model.get_value(batch_states)

            loss_value = (v_pred - batch_q_values)**2
            loss_value = 0.5 * loss_value.mean()

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

    
    def update_actor(self, batch_size, num_epoch):
        self.gpu_model.train()

        optimizer = optim.Adam(self.gpu_model.parameters(), lr=self.lr)

        storage = self.storage
        gpu_model = self.gpu_model
        model_old = self.model_old
        params_clip = self.params.clip
    
        for k in range(num_epoch):
            batch_states, batch_actions, batch_q_values, batch_log_probs = storage.actor_sample(batch_size)
    
            batch_q_values = batch_q_values
    
            with torch.no_grad():
                v_pred_old = gpu_model.get_value(batch_states)

            batch_advantages = (batch_q_values - v_pred_old)
    
            probs, mean_actions = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
            probs_old = batch_log_probs
            ratio = (probs - (probs_old)).exp()
            ratio = ratio.unsqueeze(1)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1-params_clip, 1+params_clip) * batch_advantages
            loss_clip = -(torch.min(surr1, surr2)).mean()

            total_loss = loss_clip + 0.001 * (mean_actions**2).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if self.lr > 1e-4:
            self.lr *= 0.99
        else:
            self.lr = 1e-4
    
    def save_model(self, filename):
        torch.save(self.gpu_model.state_dict(), filename)
    
    def save_shared_obs_stas(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)
    
    def save_statistics(self, filename):
        statistics = [self.time_passed, self.num_samples, self.test_mean, self.test_std, self.noisy_test_mean, self.noisy_test_std]
        with open(filename, 'wb') as output:
            pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)
    
    def collect_samples_multithread(self):
        import time
        self.num_envs = 800 # 800
        self.start = time.time()
        self.lr = 1e-3
        self.weight = 10
        num_threads = 1 # 30
        self.num_samples = 0
        self.time_passed = 0
        score_counter = 0
        total_thread = 0
        max_samples = 80000
        self.storage = PPOStorage(self.num_inputs, self.num_outputs, max_size=max_samples)
        seeds = [
            i * 100 for i in range(num_threads)
        ]
        self.explore_noise = mp.Value("f", -2.0)
        self.base_noise = np.ones(self.num_outputs)
        self.base_noise[0:14] = 1.25
        self.base_noise[14:28] = 1.25 
        noise = self.base_noise * self.explore_noise.value
        self.model.set_noise(noise)
        self.gpu_model.set_noise(noise)
        self.env.reset()
        for iterations in range(200000):
            iteration_start = time.time()
            #    print(self.model_name)
            while self.storage.counter < max_samples:
                self.collect_samples_vec(100, noise=noise)
                #    print(f"{self.storage.counter}/{max_samples}")
            start = time.time()

            self.update_critic(max_samples//4, 20)
            self.update_actor(max_samples//4, 20)
            self.storage.clear()
    
            if (iterations+1) % 500 == 0:
                self.run_test_with_noise(num_test=2)
                self.plot_statistics()
                # plt.savefig("test.png")

            print("update policy time", time.time()-start)
            print("iteration time", iterations, time.time()-iteration_start)
    
            if (iterations+0) % 1000 == 999:
                self.save_model(self.model_name+"iter%d.pt"%(iterations))
                plt.savefig(self.model_name+"test.png")
    
        self.save_reward_stats("reward_stats.npy")
        self.save_model(self.model_name+"final.pt")
    
    def add_env(self, env):
        self.env_list.append(env)
 
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
 
if __name__ == '__main__':
    import json
    from ruamel.yaml import YAML, dump, RoundTripDumper
    from raisimGymTorch.env.bin import deepmimic
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
    from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
    
    import torch
    import torch.optim as optim
    import torch.multiprocessing as mp
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.utils.data
    from model import ActorCriticNet, ActorCriticNetMann
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    seed = 3 #8
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    
    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    # create environment from the configuration file
    env = VecEnv(deepmimic.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
    print("env_created")
    env.setTask()
    ppo = RL(env, [128, 128])
    
    ppo.base_dim = ppo.num_inputs
    
    ppo.model_name = task_path + "/stats/20221103_testwalkornveleenobs/"
    
    if not(os.path.isdir(ppo.model_name)):
        os.mkdir(ppo.model_name)
    import shutil
    shutil.copy(task_path + "/cfg.yaml", ppo.model_name + "cfg.yaml")
    

    training_start = time.time()
    ppo.collect_samples_multithread()
    print("training time", time.time()-training_start)
     
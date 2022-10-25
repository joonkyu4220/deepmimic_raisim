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
   from model import ActorCriticNet
   import os
   import numpy as np
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   from tkinter import *
   from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
   from matplotlib.figure import Figure
   import matplotlib.pyplot as plt
 
   seed = 3#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # config
   cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

   # config revise
   cfg['environment']['num_envs'] = 1
   cfg['environment']['num_threads'] = 1
   cfg['environment']['mode'] = 1

   # create environment from the configuration file
   env = VecEnv(deepmimic.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

   print("env_created")

   env.setTask()
   

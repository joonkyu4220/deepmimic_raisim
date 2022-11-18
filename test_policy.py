if __name__ == '__main__':
    import time; 
    from ruamel.yaml import YAML, dump, RoundTripDumper
    from raisimGymTorch.env.bin import deepmimic
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv

    import torch
    import torch.utils.data
    from model import ActorCriticNet
    import os
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 3#8
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)
    
    # directories 
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    model_path = task_path + "/stats/20221115_walk541"

    # config
    cfg = YAML().load(open(model_path + "/cfg.yaml", 'r'))

    # create environment from the configuration file
    env = VecEnv(deepmimic.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

    print("env_created")

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    # model = ActorCriticNetMann(num_inputs, num_outputs, [128, 128])
    model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
    model.load_state_dict(torch.load(model_path + "/iter49999.pt"))
    model.cuda()

    env.setTask()
    
    env.reset()
    
    obs = env.observe()

    max_length = 100000
    
    control_dt = float(cfg["environment"]["control_dt"])
    
    for i in range(max_length):
        with torch.no_grad():
            act = model.sample_best_actions(obs)
        obs, rew, done, _ = env.step(act)
        time.sleep(control_dt)

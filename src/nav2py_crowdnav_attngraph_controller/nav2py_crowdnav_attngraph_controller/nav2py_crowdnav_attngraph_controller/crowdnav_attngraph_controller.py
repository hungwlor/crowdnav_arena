#! /usr/bin/env python3
import torch
import torch.nn as nn
import os
import rclpy.logging
import rclpy
import importlib.util
from gym.spaces import Box
from collections import OrderedDict

from crowdnav_base.rl.networks.envs import make_vec_envs
from crowdnav_base.rl.networks.model import Policy
from crowdnav_base.evaluate_2 import evaluate

MODEL_DIR = "GST_predictor_rand/"
TEST_MODEL = '41665.pt'
logger = rclpy.logging.get_logger('nav2py_template_controller')

def rand_in_range(shape, low, high, device):
    return low + (high - low) * torch.rand(shape, device=device)

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def initialize_model():
    """
    Loads the trained policy, configuration, and creates the environment.
    """
    model_dir_temp = MODEL_DIR.rstrip('/')
    arguments_path = os.path.join(model_dir_temp, "arguments.py")
    try:
        if os.path.exists(arguments_path):
            model_arguments = load_module_from_file("arguments", arguments_path)
            get_args = getattr(model_arguments, "get_args")
            logger.info('arguments')
        else:
            from crowdnav_base.arguments import get_args
    except Exception as e:
        print("Failed to load get_args from", arguments_path, ":", e)
        from crowdnav_base.arguments import get_args

    algo_args = get_args()

    configs_path = os.path.join(model_dir_temp, "configs/config.py")
    try:
        if os.path.exists(configs_path):
            model_arguments = load_module_from_file("config", configs_path)
            Config = getattr(model_arguments, "Config")
            logger.info('config')
        else:
            from crowdnav_base.crowd_nav.configs.config import Config
    except Exception as e:
        print("Failed to load Config from", configs_path, ":", e)
        from crowdnav_base.crowd_nav.configs.config import Config

    env_config = config = Config()

    # Configure torch and random seeds
    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    torch.set_num_threads(1)
    
    # Set device to cuda:0 explicitly
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    
    # Create the evaluation environment
    env_name = algo_args.env_name
    logger.info(f'{env_name}')
    eval_dir = os.path.join(MODEL_DIR, 'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
        
    # Adjust environment config as necessary
    env_config.render_traj = False
    env_config.save_slides = False
    env_config.save_path = os.path.join(MODEL_DIR, 'social_eval', TEST_MODEL[:-3])
    ax = None
    envs = make_vec_envs(env_name, algo_args.seed, 1,
                         algo_args.gamma, eval_dir, device, allow_early_resets=True,
                         config=env_config, ax=ax, test_case=-1, pretext_wrapper=config.env.use_wrapper)
    logger.info('?')
    
    # Create and load the policy network unless using a default rule-based policy
    low = -2**63
    high = 2**63 - 2
    try:
        if config.robot.policy not in ['orca', 'social_force']:
            actor_critic = Policy(
                OrderedDict([
                    ('detected_human_num', Box(low, high, shape=(1,))),
                    ('robot_node', Box(low, high, shape=(1, 7))),
                    ('spatial_edges', Box(low, high, shape=(20, 12))),
                    ('temporal_edges', Box(low, high, shape=(1, 2))),
                    ('visible_masks', Box(low, high, shape=(20,)))
                ]),
                Box(low, high, shape=(2,)),
                base_kwargs=algo_args,
                base=config.robot.policy
            )
            logger.info('policy')
            load_path = os.path.join(MODEL_DIR, 'checkpoints', TEST_MODEL)
            actor_critic.load_state_dict(torch.load(load_path, map_location=device))
            actor_critic.base.nenv = 1
            # Ensure model is on cuda:0
            actor_critic = actor_critic.to(device)
        else:
            actor_critic = None
    except Exception as e:
        logger.info(f'{e}')
    return actor_critic, envs, device, config

actor_critic, envs, device, config = initialize_model()
STEP = 1
num_processes = 1 
if STEP == 1:
    obs = envs.reset()
    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}
        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(
            num_processes, node_num, actor_critic.base.human_node_rnn_size, device=device
        )
        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(
            num_processes, edge_num, actor_critic.base.human_human_edge_rnn_size, device=device
        )
    else:
        eval_recurrent_hidden_states = None
    eval_masks = torch.zeros(num_processes, 1, device=device)
def compute_commands(x,y,linear_x,angular_z,goal_x,goal_y,rotate_z):

    global eval_recurrent_hidden_states, eval_masks, obs
    goal_x = x - goal_x
    goal_y = y - goal_y
    obs['robot_node'] = torch.tensor([[[x,y,linear_x,angular_z,goal_x,goal_y,rotate_z]]], device=device)

    # Extract the base environment for time_limit, if needed.
    # logger.info(f'{obs}')
    if hasattr(envs.venv, 'envs'):
        baseEnv = envs.venv.envs[0].env
    else:
        baseEnv = envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    with torch.no_grad():
        value, action, _, eval_recurrent_hidden_states = actor_critic.act(
            obs, 
            eval_recurrent_hidden_states, 
            eval_masks, 
            deterministic=True)
    
    obs, rew, done, infos = envs.step(action)

    STEP = 0

    return action[0][0], action[0][1]

def main():
    num_processes = 1 

    if config.robot.policy not in ['orca', 'social_force']:
        eval_recurrent_hidden_states = {}
        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(
            num_processes, node_num, actor_critic.base.human_node_rnn_size, device=device
        )
        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(
            num_processes, edge_num, actor_critic.base.human_human_edge_rnn_size, device=device
        )
    else:
        eval_recurrent_hidden_states = None

    eval_masks = torch.zeros(num_processes, 1, device=device)

    # Extract the base environment for time_limit, if needed.
    if hasattr(envs.venv, 'envs'):
        baseEnv = envs.venv.envs[0].env
    else:
        baseEnv = envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    obs = envs.reset()
    eval_masks = torch.zeros((1, 1), device=device)
    with torch.no_grad():
        value, action, _, _ = actor_critic.act(
            obs, 
            eval_recurrent_hidden_states, 
            eval_masks, 
            deterministic=True)
        
    obs, rew, done, infos = envs.step(action)
    
    print(action[0][0], action[0][1], rew, done)


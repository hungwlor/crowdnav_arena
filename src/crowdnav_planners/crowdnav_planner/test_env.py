#! /usr/bin/env python3

from geometry_msgs.msg import TwistStamped, PoseStamped
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from importlib import import_module
from rclpy.clock import Clock
from rclpy.node import Node
import rclpy.logging
import rclpy
import importlib.util
from gym.spaces import Box
from collections import OrderedDict
import numpy as np
from crowdnav_base.rl.networks.envs import make_vec_envs
from crowdnav_base.rl.networks.model import Policy
from crowdnav_base.crowd_sim import *


MODEL_DIR = '/home/sora/colcon_ws/src/CrowdNav_Prediction_AttnGraph/crowdnav_base/trained_models/GST_predictor_rand'
TEST_MODEL = '41665.pt'
logger = rclpy.logging.get_logger('controller_server')

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
    This function is called once on the first invocation of compute_velocity_commands_override.
    """
    # Remove trailing slash if present
    model_dir_temp = MODEL_DIR.rstrip('/')
    # logger.info(f'{model_dir_temp}')
    arguments_path = os.path.join(model_dir_temp, "arguments.py")
    try:
        if os.path.exists(arguments_path):
            model_arguments = load_module_from_file("arguments", arguments_path)
            get_args = getattr(model_arguments, "get_args")
            logger.info('arguments')
        else:
            from arguments import get_args
    except Exception as e:
        print("Failed to load get_args from", arguments_path, ":", e)
        from arguments import get_args

    algo_args = get_args()

	# import config class from saved directory
	# if not found, import from the default directory
    configs_path = os.path.join(model_dir_temp, "configs/config.py")
    try:
        if os.path.exists(configs_path):
            model_arguments = load_module_from_file("config", configs_path)
            Config = getattr(model_arguments, "Config")
            logger.info('config')
        else:
            from crowd_nav.configs.config import Config
    except Exception as e:
        print("Failed to load Config from", configs_path, ":", e)
        from crowd_nav.configs.config import Config

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
    device = "cpu"
    
    logger.info('dkmm torch')
    # Create the evaluation environment
    env_name = algo_args.env_name
    logger.info(f'{env_name}')
    eval_dir = os.path.join(MODEL_DIR, 'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
        
    # Adjust environment config as necessary
    env_config = config
    env_config.render_traj = False
    env_config.save_slides = False
    env_config.save_path = os.path.join(MODEL_DIR, 'social_eval', TEST_MODEL[:-3])
    ax = None
    logger.info('?')
    # Create and load the policy network unless using a default rule-based policy
    low = -2**63
    high = 2**63-2
    try:
        if config.robot.policy not in ['orca', 'social_force']:
            actor_critic = Policy(
                OrderedDict([('detected_human_num', Box(low,high,shape=(1,))), ('robot_node', Box(low,high,shape=(1, 7))), ('spatial_edges', Box(low,high,shape=(20, 12))), ('temporal_edges', Box(low,high,shape=(1, 2))), ('visible_masks', Box(low,high,shape=(20,)))]),
                Box(low,high,shape=(2,)),
                base_kwargs=algo_args,
                base=config.robot.policy)
            logger.info('policy')
            load_path = os.path.join(MODEL_DIR, 'checkpoints', TEST_MODEL)
            actor_critic.load_state_dict(torch.load(load_path, map_location=device))
            actor_critic.base.nenv = 1
            nn.DataParallel(actor_critic).to(device)
        else:
            actor_critic = None
    except Exception as e:
        logger.info(f'{e}')
    logger.info('model')
    return actor_critic, None, device, config

actor_critic, _, device, config = initialize_model()

def main():
    # # print(actor_critic)
    # observation = {
    #     'robot_node': torch.rand((1, 1, 7), device=device),
    #     'spatial_edges': torch.rand((1, 20, 12), device=device),
    #     'temporal_edges': torch.rand((1, 1, 2), device=device),
    #     'visible_masks': torch.rand((1, 20), device=device),
    #     'detected_human_num': torch.rand((1, 1), device=device),
    # }
    # eval_recurrent_hidden_states = {
    #     'human_node_rnn': torch.rand((1, 1, 128), device=device),
    #     'human_human_edge_rnn': torch.rand((1, 21, 256), device=device),
    # }
    observation = {
    'robot_node': rand_in_range((1, 1, 7), -6, 6, device),
    'spatial_edges': rand_in_range((1, 20, 12), -6, 15, device),
    'temporal_edges': rand_in_range((1, 1, 2), 0, 0, device),
    'visible_masks': torch.randint(0, 2, (1, 20), device=device, dtype=torch.bool),
    'detected_human_num': torch.tensor([[0]], device=device),
    }

    eval_recurrent_hidden_states = {
    'human_node_rnn': rand_in_range((1, 1, 128), 0, 0, device),
    'human_human_edge_rnn': rand_in_range((1, 21, 256), 0, 0, device),
    }

    eval_masks = torch.zeros((1, 1), device=device)
    with torch.no_grad():
        value,action,_,_ = actor_critic.act(observation,eval_recurrent_hidden_states,eval_masks,deterministic=True)
    print(action[0][0].numpy())
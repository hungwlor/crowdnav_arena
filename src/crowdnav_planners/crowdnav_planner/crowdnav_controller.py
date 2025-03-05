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
import logging

# Import your original modules
from crowdnav_base.rl.networks.model import Policy
from crowdnav_base.crowd_sim import *

# Global variable to hold our initialized model data
_model_data = None
goal_pose = PoseStamped()
position_all = []
logger = rclpy.logging.get_logger('controller_server')
# Configuration constants (adjust these as needed)
MODEL_DIR = '/home/sora/colcon_ws/src/CrowdNav_Prediction_AttnGraph/crowdnav_base/trained_models/GST_predictor_rand'
TEST_MODEL = '41665.pt'

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
        else:
            actor_critic = None
    except Exception as e:
        logger.info(f'{e}')
    logger.info('model')
    return actor_critic, None, device, config

actor_critic,_,device,config = initialize_model()
observation = {
'robot_node': rand_in_range((1, 1, 7), -6, 6, device),
'spatial_edges': rand_in_range((1, 20, 12), -6, 15, device),
'temporal_edges': rand_in_range((1, 1, 2), 0, 0, device),
'visible_masks': torch.randint(0, 2, (1, 20), device=device, dtype=torch.bool),
'detected_human_num': torch.tensor([[5]], device=device),
}

eval_recurrent_hidden_states = {
'human_node_rnn': rand_in_range((1, 1, 128), -1, 1, device),
'human_human_edge_rnn': rand_in_range((1, 21, 256), 0, 0, device),
}

eval_masks = torch.zeros((1, 1), device=device)
with torch.no_grad():
    value, action, _, _ = actor_critic.act(observation, eval_recurrent_hidden_states, eval_masks, deterministic=True)
# logger.info(f"Action: {action}")

def compute_velocity_commands_override(occupancy_grid, pose, twist):
    """
    Computes velocity commands using the loaded model and returns them as a TwistStamped message.
    
    Parameters:
      occupancy_grid - sensor data representing the environment (placeholder).
      pose           - the current pose of the robot.
      twist          - the current twist (velocity) of the robot.
      
    Returns:
      cmd_vel        - a geometry_msgs.msg.TwistStamped message containing the computed commands.
    """    # logger.info('dkmm')

    logger.info('dkmm')
    # ----- Convert sensor data to a model observation -----
    # NOTE: You must implement your own conversion logic here based on your sensor inputs.
    # The following is a placeholder that simply resets the environment.
    observation = {
    'robot_node': rand_in_range((1, 1, 7), -6, 6, device),
    'spatial_edges': rand_in_range((1, 20, 12), -6, 15, device),
    'temporal_edges': rand_in_range((1, 1, 2), 0, 0, device),
    'visible_masks': torch.randint(0, 2, (1, 20), device=device, dtype=torch.bool),
    'detected_human_num': torch.tensor([[5]], device=device),
    }

    eval_recurrent_hidden_states = {
    'human_node_rnn': rand_in_range((1, 1, 128), -1, 1, device),
    'human_human_edge_rnn': rand_in_range((1, 21, 256), 0, 0, device),
    }

    eval_masks = torch.zeros((1, 1), device=device)
    logger.info(f'{type(actor_critic)}')
    logger.info(f'obs: {observation}')
    try:
        logger.info('start')
        value,action,_,_ = actor_critic.act(observation,eval_recurrent_hidden_states,eval_masks,deterministic=True)
    except Exception as e:
        logger.warning(f"Error in model prediction: {e}")
    cmd_vel = TwistStamped()
    cmd_vel.header = pose.header
    cmd_vel.twist.linear.x = 2.0
    cmd_vel.twist.angular.z = 0.0
    # return cmd_vel
    # try:
    #     logging.warning('start')
    #     with torch.no_grad():
    #         value, action, _, _ = actor_critic.act(observation, eval_recurrent_hidden_states, eval_masks, deterministic=True)
    #     logger.info(f"Action: {action}")
    #     logging.warning(f"Action: {action}")
    cmd_vel.header = pose.header
    logger.info(f'{action}'
    )
    logger.info(f'{action[0][0].numpy()}')
    logger.info(f'{action[0][1].numpy()}')
    linear_x = action[0][0].numpy()
    angular_z = action[0][1].numpy()    
    try:
        cmd_vel.twist.linear.x = float(linear_x)
        cmd_vel.twist.angular.z = float(angular_z)
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
    # Assuming the action has at least two components: [linear_velocity, angular_velocity]
    # If action is batched, select the first sample.
    # logger.info(f'{cmd_vel}')
    return cmd_vel
    # except Exception as e:
    #     logger.error(f"Error in model prediction: {e}")
    #     return cmd_vel

    # raise ValueError(action)
    # Create a ROS2 TwistStamped message with the action results
    

def handleGlobalPlan(global_path):
    position_x = []
    position_y = []
    i=0
    while(i <= len(global_path.poses)-1):
        position_x.append(global_path.poses[i].pose.position.x)
        position_y.append(global_path.poses[i].pose.position.y)
        i=i+1
    position_all = [list(double) for double in zip(position_x,position_y)]
    
    return position_all

def setPath(global_plan):
    global goal_pose 
    goal_pose = global_plan.poses[-1]
    global position_all
    position_all = handleGlobalPlan(global_plan)
    return

def setSpeedLimit(speed_limit, is_percentage):
    return

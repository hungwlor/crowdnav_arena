from geometry_msgs.msg import TwistStamped
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from importlib import import_module
from rclpy.clock import Clock

# Import your original modules
from crowdnav_base.rl.networks.envs import make_vec_envs
from crowdnav_base.rl.networks.model import Policy
from crowdnav_base.crowd_sim import *

# Global variable to hold our initialized model data
_model_data = None

# Configuration constants (adjust these as needed)
MODEL_DIR = '/home/sora/colcon_ws/src/CrowdNav_Prediction_AttnGraph/crowdnav_base/trained_models/GST_predictor_rand'
TEST_MODEL = '41665.pt'

def initialize_model():
    """
    Loads the trained policy, configuration, and creates the environment.
    This function is called once on the first invocation of compute_velocity_commands_override.
    """
    # Remove trailing slash if present
    model_dir_temp = MODEL_DIR.rstrip('/')
    
    # Import model arguments
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.arguments'
        model_arguments = import_module(model_dir_string)
        get_args = getattr(model_arguments, 'get_args')
    except Exception as e:
        print('Failed to load get_args from', MODEL_DIR + '/arguments.py')
        from crowdnav_base.arguments import get_args
    
    algo_args = get_args()
    
    # Import configuration
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
    except Exception as e:
        print('Failed to load Config from', MODEL_DIR)
        from crowdnav_base.crowd_nav.configs.config import Config
    config = Config()
    
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
    device = torch.device("cuda" if algo_args.cuda else "cpu")
    
    # Create the evaluation environment
    env_name = algo_args.env_name
    eval_dir = os.path.join(MODEL_DIR, 'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    # Adjust environment config as necessary
    env_config = config
    env_config.render_traj = False
    env_config.save_slides = False
    env_config.save_path = os.path.join(MODEL_DIR, 'social_eval', TEST_MODEL[:-3])
    ax = None
    
    envs = make_vec_envs(env_name, algo_args.seed, 1,
                         algo_args.gamma, eval_dir, device,
                         allow_early_resets=True, config=env_config, ax=ax,
                         test_case=-1, pretext_wrapper=config.env.use_wrapper)
    
    # Create and load the policy network unless using a default rule-based policy
    if config.robot.policy not in ['orca', 'social_force']:
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy)
        load_path = os.path.join(MODEL_DIR, 'checkpoints', TEST_MODEL)
        actor_critic.load_state_dict(torch.load(load_path, map_location=device))
        actor_critic.base.nenv = 1
        actor_critic = nn.DataParallel(actor_critic).to(device)
    else:
        actor_critic = None

    return actor_critic, envs, device, config, algo_args

def compute_velocity_commands_override(occupancy_grid, pose, twist):
    """
    Computes velocity commands using the loaded model and returns them as a TwistStamped message.
    
    Parameters:
      occupancy_grid - sensor data representing the environment (placeholder).
      pose           - the current pose of the robot.
      twist          - the current twist (velocity) of the robot.
      
    Returns:
      cmd_vel        - a geometry_msgs.msg.TwistStamped message containing the computed commands.
    """
    global _model_data
    if _model_data is None:
        _model_data = initialize_model()
    
    actor_critic, envs, device, config, algo_args = _model_data

    # ----- Convert sensor data to a model observation -----
    # NOTE: You must implement your own conversion logic here based on your sensor inputs.
    # The following is a placeholder that simply resets the environment.
    observation = envs.reset()

    # ----- Compute the action using the loaded policy -----
    with torch.no_grad():
        # The act() function is expected to return (value, action, log_prob, hidden_state)
        value, action, _, _ = actor_critic.act(observation, None, None, deterministic=True)
    
    # Create a ROS2 TwistStamped message with the action results
    cmd_vel = TwistStamped()
    clock = Clock()
    cmd_vel.header.stamp = clock.now().to_msg()
    cmd_vel.header.frame_id = "base_link"  # Adjust frame_id if needed
    
    # Assuming the action has at least two components: [linear_velocity, angular_velocity]
    # If action is batched, select the first sample.
    if action.dim() > 1:
        action = action[0]
    cmd_vel.twist.linear.x = float(action[0]) if action.nelement() > 0 else 0.0
    cmd_vel.twist.angular.z = float(action[1]) if action.nelement() > 1 else 0.0

    return cmd_vel

def set_plan_override(global_plan):
    """
    Overrides the global plan.
    
    Parameters:
      global_plan - the current global plan.
      
    Returns:
      None
    """
    # Integration with your own code here
    return

def set_speed_limit_override(speed_limit, is_percentage):
    """
    Overrides the speed limit settings.
    
    Parameters:
      speed_limit  - the speed limit value.
      is_percentage- flag indicating if the speed limit is a percentage.
      
    Returns:
      None
    """
    # Integration with your own code here
    return

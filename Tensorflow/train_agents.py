import logging
import os

import numpy as np
import tensorflow as tf
from stable_baselines3.common.utils import set_random_seed
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from sklearn.model_selection import train_test_split

from utils.utils import load_data, read_config, denorm
from RL_Tools.Environments.LDEnvs import BoxActionLDEnv, DiscreteActionLDEnv
from utils.load_ae_models import load_ae_models
import yaml
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from datetime import datetime
from stable_baselines3 import DDPG, TD3, PPO, A2C, DQN, HER, SAC
from RL_Tools.rl_uitls.plot_curves import plot_results
from RL_Tools.rl_uitls.SaveBestModelCallback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback
from utils.fit_geom import fit_geom, plot_fit_curve, calc_y_distance


logging.basicConfig(level=logging.INFO)
logging.info(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

try:
    for gpu in tf.config.experimental.list_physical_devices(
            "GPU"): tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
except Exception as e:
    logging.info(e)


possible_box_action_agents={
    'A2C':A2C,
    'DDPG': DDPG,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3,
}

possible_discrete_action_agents={
    'A2C': A2C,
    'DQN': DQN,
    'PPO': PPO,
}

def pred_latent_data(ae_models, train_data, test_data):
    train_latent = ae_models['encoder'].predict(train_data)
    test_latent = ae_models['encoder'].predict(test_data)
    return train_latent, test_latent

def gen_data_for_envs(rl_config):
    # Load data
    normed_geometries, normed_global_variables, scaler_globals, min_y, max_y = load_data('dataset_complete')

    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(
        normed_geometries, normed_global_variables, test_size=0.1, shuffle=True, random_state=22)
    print('train data', train_data.shape)
    print('test data', test_data.shape)
    print('train_labels', train_labels.shape)
    print('test_labels', test_labels.shape)


    # Load models for the environment
    ae_models = load_ae_models(rl_config)

    # insert function for the denormalization
    ae_models['scaler_globals'] = scaler_globals
    ae_models['scaler_geom'] = {
        'min_y': min_y,
        'max_y': max_y
    }
    ae_models['denorm_geom'] = denorm
    ae_models['fit_geom'] = fit_geom
    ae_models['plot_fit_curve'] = plot_fit_curve
    ae_models['calc_y_distance'] = calc_y_distance

    # Generate Latent data for Training and Test
    train_latent, test_latent = pred_latent_data(ae_models, train_data, test_data)

    data_env_train = {'cod': train_latent,
                      'origin_geom': train_data,
                      'origin_global_variables': train_labels,
                      }

    data_env_test = {'cod': test_latent,
                     'origin_geom': test_data,
                     'origin_global_variables': test_labels,
                     }

    return ae_models,data_env_train, data_env_test

def wrap_env(gym_env, log_path=None):
    if log_path is not None:
        gym_env = Monitor(gym_env, log_path)
    else:
        gym_env = Monitor(gym_env)
    env = DummyVecEnv([lambda: gym_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    env = VecFrameStack(env, n_stack=4)
    return env

if __name__ == '__main__':
    with tf.device('/CPU:0'):
        rl_config = read_config('RL_Tools/rl_config.yaml')
        print('rl_config', read_config)

        all_resuls_path = 'results/RL_results/'
        if rl_config['Re_Training']['active']:
            agent_name= rl_config['Re_Training']['model_path']
        else:
            agent_name = 'agent_'+datetime.now().strftime("%d_%m_%Y %H_%M_%S")

        results_path = all_resuls_path+agent_name
        CHECK_FOLDER = os.path.isdir(results_path)

        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(results_path)
            print("created folder : ", results_path)

        # save the current rl_config file in results path
        with open(results_path+'/rl_config.yml', 'w') as yaml_file:
            yaml.dump(rl_config, yaml_file, default_flow_style=False)

        # Create data for environments
        ae_models,data_env_train, data_env_test = gen_data_for_envs(rl_config)

        if rl_config['action_space']=='BoxSpaceAction':
            gym_env_train = BoxActionLDEnv(ae_models, data_env_train, rl_config)
            gym_env_eval = BoxActionLDEnv(ae_models, data_env_test, rl_config)

            agent = possible_box_action_agents[rl_config['type_agent']]
        else:
            gym_env_train = DiscreteActionLDEnv(ae_models, data_env_train, rl_config)
            gym_env_eval = BoxActionLDEnv(ae_models, data_env_test, rl_config)

            agent = possible_discrete_action_agents[rl_config['type_agent']]

        # Wrapper envs
        env_train = wrap_env(gym_env_train, log_path= results_path)
        env_eval = wrap_env(gym_env_eval)

        # create agent
        model = agent(rl_config['agent_config']['policy_network'], env_train, verbose=0,
                      tensorboard_log='logs/tb_stable_baseline_logs/' + agent_name)

        # if rl_config['Re_Training']['active'] == True:
        #     model = agent.load(results_path+'/best_model_agent.zip',env_train, tensorboard_log='logs/tb_stable_baseline_logs/' + agent_name)


        # Create Callbacks
        callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=results_path)
        # eval_callback = EvalCallback(env_eval, best_model_save_path=results_path,
        #                              log_path=results_path, eval_freq=1000,
        #                              deterministic=True, render=False)

        # evaluate model before training


        # Train
        model.learn(total_timesteps=int(rl_config['total_timesteps']), callback=callback, reset_num_timesteps=not rl_config['Re_Training']['active'])





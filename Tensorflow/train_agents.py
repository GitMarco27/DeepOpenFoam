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


def gen_data_for_envs(rl_config):
    # Load data
    normed_geometries, normed_global_variables, scaler_globals, min_y, max_y = load_data('dataset')

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

    # Generate Latent data for Training and Test
    train_latent = ae_models['encoder'].predict(train_data)
    test_latent = ae_models['encoder'].predict(test_data)

    data_env_train = {'cod': train_latent,
                      'origin_geom': train_data,
                      'origin_global_variables': train_labels,
                      }

    data_env_test = {'cod': test_latent,
                     'origin_geom': test_data,
                     'origin_global_variables': test_labels,
                     }

    return ae_models,data_env_train, data_env_test

if __name__ == '__main__':
    results_path = 'results/RL_results/agent_'+datetime.now().strftime("%d_%m_%Y %H_%M_%S")
    CHECK_FOLDER = os.path.isdir(results_path)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(results_path)
        print("created folder : ", results_path)

    rl_config = read_config('RL_Tools/rl_config.yaml')
    print('rl_config', read_config)

    # save the current rl_config file in results path
    with open(results_path+'/rl_config.yml', 'w') as yaml_file:
        yaml.dump(rl_config, yaml_file, default_flow_style=False)

    # Create data for environments
    ae_models,data_env_train, data_env_test = gen_data_for_envs(rl_config)

    if rl_config['type_agent']=='BoxActionLDEnv':
        gym_env_train = BoxActionLDEnv(ae_models, data_env_train, rl_config)
        gym_env_eval = BoxActionLDEnv(ae_models, data_env_test, rl_config)
    else:
        gym_env_train = DiscreteActionLDEnv(ae_models, data_env_train, rl_config)
        gym_env_eval = BoxActionLDEnv(ae_models, data_env_test, rl_config)

    # Wrapper envs




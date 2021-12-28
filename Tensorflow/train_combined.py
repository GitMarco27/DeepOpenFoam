import argparse
import os
from utils.chamfer_distance import chamfer_distance
from itertools import product
import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict, namedtuple
import tensorflow as tf

from Tensorflow.utils.concatenate_sides import concatenate_sides
from Tensorflow.utils.PointNetAE import  create_pointnet_ae

def scale_y_points(x):
    # @TODO: how to denormalize data ?
    x_norm = x.copy()
    x_y = x_norm[:, :, 1].reshape(-1, 1)
    min_v_y = min(x_y) + 0.2 * min(x_y)  # @TODO: beky => check this
    max_v_y = max(x_y) + 0.2 * max(x_y)

    x_scaled_y = (x_y - min_v_y / (max_v_y - min_v_y))

    x_norm[:, :, 1] = x_scaled_y.reshape(x[:, :, 1].shape)
    return x_norm, min_v_y, max_v_y


def load_data(path: str = 'dataset'):
    pressure_side = np.load(f'{path}/pressure_side.npy')
    suction_side = np.load(f'{path}/suction_side.npy')
    data = concatenate_sides(suction_side, pressure_side)
    # escludo il campo di pressione dai dati
    data[:, :, [3, 4]] = data[:, :, [4, 3]]
    data = data[:, :, :4]
    print('Data shape: ', data.shape)

    global_variables_ = pd.read_csv(f'{path}/coefficients_clean.csv')
    global_variables_ = global_variables_.iloc[:, -2:].to_numpy()
    scaler_globals_ = MinMaxScaler()

    normed_global_variables_ = scaler_globals_.fit_transform(global_variables_)

    normed_geometries_, min_value_y, max_value_x = scale_y_points(data)

    return normed_geometries_, normed_global_variables_, scaler_globals_


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def handle_results_path():
    if args.clear:
        if os.path.exists(args.results_path):
            os.system(f'rm -r {args.results_path}')
            os.system(f'rm -r {args.log_path}')
            os.mkdir(args.results_path)
            os.mkdir(args.log_path)
        else:
            os.mkdir(args.results_path)
            os.mkdir(args.log_path)
    else:
        if os.path.exists(args.results_path):
            raise ValueError(f'{args.results_path} already exists')
        else:
            os.mkdir(args.results_path)
            os.mkdir(args.log_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training, validation and testing of AE and 3D regressor')

    # Required positional argument
    parser.add_argument('--data_path', type=str,
                        default='dataset',
                        help='dataset path')

    parser.add_argument('--results_path', type=str,
                        default='results',
                        help='results path')
    parser.add_argument('--log_path', type=str,
                        default='logs',
                        help='log path')
    parser.add_argument('--clear', type=bool,
                        default=True,  # @TODO: change
                        help='delete results path if exists')

    args = parser.parse_args()

    normed_geometries, normed_global_variables, scaler_globals = load_data(args.data_path)

    train_data, test_data, train_labels, test_labels = train_test_split(
        normed_geometries, normed_global_variables, test_size=0.2, random_state=22)

    params = OrderedDict(
        lr=[.01, .001],
        batch_size=[32, ],
        epochs=[5, ],
        optimizer=['Adam', 'RMSprop', 'adadelta'],
        type_decoder= ['dense', 'cnn'],
        architectural_parameters = [[False, 0], [True, 1], [True, 5], [True, 10], [True, 15], [True, 20]], # [is_variational, beta],
        encoding_size = [5,10,20,50,100],
        ort_reg_bools = [[False,False], [True, False], [True, True]], #[feature_transform, orto_reg]
        reg_drop_out_value = [0., 0.1, 0.3 , 0.5]
    )

    handle_results_path()

    for run in tqdm.tqdm(RunBuilder.get_runs(params)):

        print('--- New run detected ---')

        for key in run._asdict():
            print(f'{key}: {run._asdict()[key]}')

        run_path = os.path.join(args.results_path, str(run).replace(" ", ""))

        print(f'\n Creating results path: {run_path}')
        os.mkdir(run_path)

        log_dir = os.path.join(args.log_path, str(run).replace(" ", ""))
        os.mkdir(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model = create_pointnet_ae(run)

        model.summary() if model is not None else None

        optimizer = tf.keras.optimizers.get(run.optimizer)
        optimizer.learning_rate = run.lr

        model.compile(
            optimizer=optimizer,
            loss=[chamfer_distance, tf.keras.metrics.mean_squared_error],
            metrics=None,
        )




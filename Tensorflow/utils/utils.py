import os
import pandas as pd
from matplotlib import pyplot as plt

from . import concatenate_sides
from itertools import product
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict, namedtuple


def scale_y_points(x):
    x_norm = x.copy()
    x_y = x_norm[:, :, 1].reshape(-1, 1)
    min_v_y = min(x_y) + 0.2 * min(x_y)
    max_v_y = max(x_y) + 0.2 * max(x_y)

    x_scaled_y = (x_y - min_v_y / (max_v_y - min_v_y))

    x_norm[:, :, 1] = x_scaled_y.reshape(x[:, :, 1].shape)
    return x_norm, min_v_y, max_v_y


def load_data(path: str = 'dataset'):
    pressure_side = np.load(f'{path}/pressure_side.npy')
    suction_side = np.load(f'{path}/suction_side.npy')
    data = concatenate_sides.concatenate_sides(suction_side, pressure_side)
    # escludo il campo di pressione dai dati
    data[:, :, [3, 4]] = data[:, :, [4, 3]]
    data = data[:, :, :4]
    print('Data shape: ', data.shape)

    global_variables_ = pd.read_csv(f'{path}/coefficients_clean.csv')
    global_variables_ = global_variables_.iloc[:, -2:].to_numpy()
    scaler_globals_ = MinMaxScaler()

    normed_global_variables_ = scaler_globals_.fit_transform(global_variables_)

    normed_geometries_, min_value_y, max_value_y = scale_y_points(data)

    return normed_geometries_, normed_global_variables_, scaler_globals_, min_value_y, max_value_y


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def handle_results_path(args):
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


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2), dist_2


def fit_bezier(o: int,
               data: np.ndarray,

               ):
    from scipy.special import comb as n_over_k
    Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
    bezier_coefficients = lambda ts, order: [[Mtk(order - 1, t, k) for k in range(order)] for t in ts]

    point = data[np.argmin(data[:, 0])]
    new_data = []
    new_data.append(point)
    tmp_data = data[data[:, 0] > point[0]]

    for j in range(data.shape[0]):
        if len(tmp_data) == 0:
            break
        closest, _ = closest_node(point, tmp_data)
        closest = tmp_data[closest]
        point = closest
        new_data.append(point)
        tmp_data = tmp_data[tmp_data[:, 0] > point[0]]

    new_data = np.asarray(new_data)
    tData = np.linspace(0., 1., new_data.shape[0])

    Pseudoinverse = np.linalg.pinv(bezier_coefficients(tData, o))
    control_points = Pseudoinverse.dot(new_data)
    bezier = np.array(bezier_coefficients(tData, o)).dot(control_points)

    return bezier, control_points


def denorm(x_norm, min_v_y, max_v_y):
    x = x_norm.copy()
    x_scaled_y = x_norm[:, :, 1].reshape(-1, 1)

    x_y = x_scaled_y * (max_v_y - min_v_y) + min_v_y

    x[:, :, 1] = x_y.reshape(x[:, :, 1].shape)
    return x


def plot_airfoil(airfoil, cl=0, cd=0):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    upper = airfoil[np.round(airfoil[:, -1], 0) == 0][:, :2]
    lower = airfoil[np.round(airfoil[:, -1], 0) == 1][:, :2]

    upper_, upper_control_points = fit_bezier(12, upper)
    lower_, lower_control_points = fit_bezier(12, lower)

    axs[0].scatter(upper[:, 0], upper[:, 1], s=10, edgecolor='k', label='upper')
    axs[0].scatter(lower[:, 0], lower[:, 1], s=10, edgecolor='k', label='lower')

    axs[0].set_xlim([-0.1, 1.1])
    axs[0].set_ylim([-0.6, 0.6])

    axs[0].set_title('Cl: %.4f - Cd : %.4f - eta: %.4f' % (cl, cd, cl / cd))
    axs[0].legend()

    axs[1].plot(upper_[:, 0],
                upper_[:, 1], 'b--', label='fit_upper', fillstyle='none')
    axs[1].plot(lower_[:, 0],
                lower_[:, 1], 'r--', label='fit_lower', fillstyle='none')

    # axs[1].scatter(upper[:, 0], upper[:, 1], s=10, edgecolor='k', label='upper')
    # axs[1].scatter(lower[:, 0], lower[:, 1], s=10, edgecolor='k', label='lower')
    # axs[1].plot(upper_control_points[:,0],
    #             upper_control_points[:,1], 'ko:', fillstyle='none')

    axs[1].set_xlim([-0.1, 1.1])
    axs[1].set_ylim([-0.6, 0.6])
    axs[1].legend()

    plt.show()

    return upper_, lower_

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb as n_over_k
import matplotlib.pyplot as plt

def fit_pol(x, *coeffs):
    y = np.polyval(coeffs, x)
    return y

def fit_bezier(x, *coeffs):
    o = int(len(coeffs) / 2)
    Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
    bezier_coefficients = lambda ts, order: [[Mtk(order - 1, t, k) for k in range(order)] for t in ts]
    control_points = np.array(coeffs).reshape(-1, 2)
    bezier = np.array(bezier_coefficients(x, o)).dot(control_points)
    return bezier[:, 1]


def cal_error(x, y, fit, coeffs):
    x_line = x
    # calculate the output for the range
    y_line = fit(x_line, *coeffs)

    fitting_error = np.sqrt((y - y_line) ** 2).mean()
    return fitting_error

def fit_curve(x, y, order: int = 10, function_string: str = 'fit_pol'):

    if function_string == 'fit_pol':
        fit = fit_pol
    else:
        fit = fit_bezier

    if order > len(x):
        order = len(x)-1

    p0 = np.ones(order)
    popt, pcov = curve_fit(fit, x, y, p0=p0, maxfev = 3000)

    # summarize the parameter values
    coeffs = popt

    x_line = np.linspace(start=min(x), stop=max(x), num=len(x))

    # calculate the output for the range
    y_line = fit(x_line, *coeffs)

    fitting_error = cal_error(x,y, fit, coeffs)

    return x_line, y_line, popt, pcov, fitting_error


def fit_geom(geom, order: int = 10, sub_geom: int = 2, function_string: str = 'fit_pol'):
    upper_geom = geom[geom[:, 3] <= 0.5]
    lower_geom = geom[geom[:, 3] > 0.5]

    delta_lower = (lower_geom[:, 0].max() - lower_geom[:, 0].min()) / sub_geom
    delta_upper = (upper_geom[:, 0].max() - upper_geom[:, 0].min()) / sub_geom

    fitting_error = 0.

    curv = {
        'lower': {
            'all_curves': [],
            'all_models': []
        },
        'upper': {
            'all_curves': [],
            'all_models': []
        }
    }

    for i in range(1, sub_geom + 1):
        u_g_i = upper_geom[(upper_geom[:, 0] > (i - 1) * delta_upper) & (upper_geom[:, 0] < i * delta_upper)]
        l_g_i = lower_geom[(lower_geom[:, 0] > (i - 1) * delta_lower) & (lower_geom[:, 0] < i * delta_lower)]

        x_h = u_g_i[:, 0]
        y_h = u_g_i[:, 1]
        x_l = l_g_i[:, 0]
        y_l = l_g_i[:, 1]

        x_line_l, y_line_l, popt_l, pcov_l, fitting_error_lower = fit_curve(x_l, y_l, order=order,
                                                                            function_string=function_string)
        x_line_h, y_line_h, popt_h, pcov_h, fitting_error_upper = fit_curve(x_h, y_h, order=order,
                                                                            function_string=function_string)

        fitting_error = fitting_error + 1 / 2 * (fitting_error_lower + fitting_error_upper)

        curv['lower']['all_curves'].append(np.array([x_line_l, y_line_l]).T)
        curv['upper']['all_curves'].append(np.array([x_line_h, y_line_h]).T)

        curv['lower']['all_models'].append(popt_l)
        curv['upper']['all_models'].append(popt_h)

    fitting_error = fitting_error / sub_geom
    return curv, fitting_error


def plot_fit_curve(curvs, color='black'):
    for curv in curvs['lower']['all_curves']:
        plt.plot(curv[:, 0], curv[:, 1], color=color)
    for curv in curvs['upper']['all_curves']:
        plt.plot(curv[:, 0], curv[:, 1] , color=color)

def calc_y_distance(fitting_curve, fit_params, isPlotTrue: bool = False):
    model_lower = fitting_curve['lower']['all_models'][0]
    model_upper = fitting_curve['upper']['all_models'][0]

    x_line = np.linspace(start=fit_params['th_x'], stop=fit_params['th_x_max'], num=150)

    y_lower = fit_pol(x_line, *model_lower.tolist())
    y_upper = fit_pol(x_line, *model_upper.tolist())

    y_distance = y_upper - y_lower

    min_distance = y_distance.min()

    abnormal_points = min_distance <= fit_params['y_distance_th']

    if abnormal_points:
        x_line_th = x_line[y_distance < fit_params['y_distance_th']]
        min_anomaly_x = x_line_th.min()
        max_anomaly_x = x_line_th.max()

        edges_anomalous_region = {
            'min_x': min_anomaly_x,
            'max_x': max_anomaly_x
        }
    else:
        edges_anomalous_region = {}

    if isPlotTrue:
        fig, ax = plt.subplots(2, figsize=(10, 10))
        ax[0].hist(y_distance, bins=100)
        ax[0].set_title('Distance Histogram')

        ax[1].plot(x_line, y_lower, c='blue')
        ax[1].plot(x_line, y_upper, c='blue')
        if abnormal_points:
            ax[1].axvspan(min_anomaly_x, max_anomaly_x, facecolor='r', alpha=0.5)
        ax[1].set_ylim([-0.35, 0.35])
        ax[1].set_xlim([-0.02, 1.02])
        ax[1].set_title('Geometry')
        plt.show()
    return abnormal_points, edges_anomalous_region
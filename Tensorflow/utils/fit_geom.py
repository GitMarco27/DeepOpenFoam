import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb as n_over_k
import matplotlib.pyplot as plt

def fit_pol(x, *coeffs):
    y = np.polyval(coeffs, x)
    return y


def fit_exp(x, b, *coeffs):
    coeffs_single_exp = np.array(coeffs).reshape(-1, 2)
    y = b
    for i in range(coeffs_single_exp.shape[0]):
        c1 = coeffs_single_exp[i, 0]
        c2 = coeffs_single_exp[i, 1]
        y = c1 * np.exp(-c2 * x)
    return y


def fit_log(x, b, *coeffs):
    coeffs_single_log = np.array(coeffs).reshape(-1, 2)
    y = b
    for i in range(coeffs_single_log.shape[0]):
        c1 = coeffs_single_log[i, 0]
        c2 = coeffs_single_log[i, 1]
        y = c1 * np.log(c2 * x)
    return y


def fit_mix_function(x, *coeffs):
    coeffs_pol = coeffs[:int(len(coeffs) / 3)]
    coeffs_exp = coeffs[int(len(coeffs) / 3):int(len(coeffs) * 2 / 3)]
    coeffs_log = coeffs[int(len(coeffs) * 2 / 3):]

    y_pol = fit_pol(x, *coeffs_pol)
    y_exp = fit_exp(x, *coeffs_exp)
    y_log = fit_log(x, *coeffs_log)

    return y_pol + y_exp + y_log


def fit_bezier(x, *coeffs):
    o = int(len(coeffs) / 2)
    Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
    bezier_coefficients = lambda ts, order: [[Mtk(order - 1, t, k) for k in range(order)] for t in ts]
    control_points = np.array(coeffs).reshape(-1, 2)
    bezier = np.array(bezier_coefficients(x, o)).dot(control_points)
    return bezier[:, 1]


def fit_curve(x, y, order: int = 10, function_string: str = 'fit_pol'):
    if function_string == 'fit_exp':
        fit = fit_exp
    elif function_string == 'fit_pol':
        fit = fit_pol
    elif function_string == 'mix':
        fit = fit_mix_function
    else:
        fit = fit_bezier

    p0 = np.ones(order)
    popt, pcov = curve_fit(fit, x, y, p0=p0)

    # summarize the parameter values
    coeffs = popt

    # x_line = np.arange(min(x), max(x), 1/len(x))
    x_line = x
    # calculate the output for the range
    y_line = fit(x, *coeffs)

    fitting_error = np.sqrt((x - x_line) ** 2 + (y - y_line) ** 2).mean()

    return x_line, y_line, popt, pcov, fitting_error


def fit_geom(geom, order: int = 10, sub_geom: int = 2, function_string: str = 'fit_pol'):
    upper_geom = geom[geom[:, 3] <= 0.5]
    lower_geom = geom[geom[:, 3] > 0.5]

    delta_lower = (lower_geom[:, 0].max() - lower_geom[:, 0].min()) / sub_geom
    delta_upper = (upper_geom[:, 0].max() - upper_geom[:, 0].min()) / sub_geom

    fitting_error = 0.

    curv = {
        'lower': {
            'all_curves': []
        },
        'upper': {
            'all_curves': []
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

    fitting_error = fitting_error / sub_geom
    return curv, fitting_error


def plot_fit_curve(curvs):
    for curv in curvs['lower']['all_curves']:
        plt.scatter(curv[:, 0], curv[:, 1], color='red')
    for curv in curvs['upper']['all_curves']:
        plt.scatter(curv[:, 0], curv[:, 1], color='red')
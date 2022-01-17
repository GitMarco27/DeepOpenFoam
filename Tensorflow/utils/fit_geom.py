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


def plot_fit_curve(curvs, color='black'):
    for curv in curvs['lower']['all_curves']:
        plt.plot(curv[:, 0], curv[:, 1], color=color)
    for curv in curvs['upper']['all_curves']:
        plt.plot(curv[:, 0], curv[:, 1] , color=color)
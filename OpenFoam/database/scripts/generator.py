import os
import numpy as np


def noise(n: int,
          eta: float = .2):
    x = (np.random.rand(n)*2 - 1) * eta
    x[0] = 0
    x[-1] = 0
    return x


if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    # Generating control points
    xs = 0
    ys = 0
    xe = 1
    ye = 0

    baseline_upper = 0.1
    baseline_lower = 0.0

    x_u = np.array([xs, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, xe])
    x_l = np.array([xs, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, xe])

    y_u = np.array([ys] + [baseline_upper] * 10 + [ye]) + noise(12, eta=0.1)
    y_l = np.array([ys] + [-baseline_lower] * 10 + [ye]) + noise(12, eta=0.15)

    y_u[1] = 0.03 if y_u[1] <= 0.03 else y_u[1]
    y_l[1] = -0.03 if y_l[1] >= -0.03 else y_l[1]
    # y_l[1] = -y_u[1]

    if os.path.exists('mesh/bez_control_points.geo'):
        # print('Deleting old parameters...')
        os.system('rm -r mesh/bez_control_points.geo')

    with open('mesh/bez_control_points.geo', 'w') as f:
        f.write(f'xs={xs};\n')
        f.write(f'ys={ys};\n')
        f.write(f'xe={xe};\n')
        f.write(f'ye={ye};\n')

        f.write(
            f'x_u = {{{x_u[0]},{x_u[1]},{x_u[2]},{x_u[3]},{x_u[4]},{x_u[5]},{x_u[6]},{x_u[7]},{x_u[8]},{x_u[9]},{x_u[10]},{x_u[11]}}};\n')
        f.write(
            f'x_l = {{{x_l[0]},{x_l[1]},{x_l[2]},{x_l[3]},{x_l[4]},{x_l[5]},{x_l[6]},{x_l[7]},{x_l[8]},{x_l[9]},{x_l[10]},{x_l[11]}}};\n')
        f.write(
            f'y_u = {{{y_u[0]},{y_u[1]},{y_u[2]},{y_u[3]},{y_u[4]},{y_u[5]},{y_u[6]},{y_u[7]},{y_u[8]},{y_u[9]},{y_u[10]},{y_u[11]}}};\n')
        f.write(
            f'y_l = {{{y_l[0]},{y_l[1]},{y_l[2]},{y_l[3]},{y_l[4]},{y_l[5]},{y_l[6]},{y_l[7]},{y_l[8]},{y_l[9]},{y_l[10]},{y_l[11]}}};\n')

    # print('--- END-SCRIPT --- ')

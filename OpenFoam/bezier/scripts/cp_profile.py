import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    parser = argparse.ArgumentParser(description='Plotting Cp profile')
    parser.add_argument('--U', type=float,
                        default=45.0,
                        help='Farfield velocity magnitude')
    parser.add_argument('--rho_inf', type=float,
                        default=1.18,
                        help='Farfield density')
    parser.add_argument('--p_inf', type=float,
                        default=0.0,
                        help='Farfield pressure')

    args = parser.parse_args()

    print('Loading data...')
    file = [item for item in os.listdir('case/postProcessing/wallPressure') if '.DS_Store' not in item][-1]
    print(f'file: {file}')
    p_p = np.loadtxt(f'case/postProcessing/wallPressure/{file}/p_p_side.raw', dtype=np.float32)
    p_s = np.loadtxt(f'case/postProcessing/wallPressure/{file}/p_s_side.raw', dtype=np.float32)

    p_p = p_p[p_p[:, 2] == 0.]
    p_s = p_s[p_s[:, 2] == 0.]

    p_s[:, 3] = (p_s[:, 3] - args.p_inf) / (0.5 * args.rho_inf * args.U ** 2)
    p_p[:, 3] = (p_p[:, 3] - args.p_inf) / (0.5 * args.rho_inf * args.U**2)


    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(p_p[:, 0], p_p[:, 3],
                label='Pressure Side',
                s=10,
                # c=p_p[:, 3],
                edgecolors='k',
                cmap='viridis'
                )
    plt.scatter(p_s[:, 0], p_s[:, 3],
                label='Suction Side',
                s=10,
                # c=p_p[:, 3],
                edgecolors='k',
                cmap='viridis'
                )
    plt.xlabel('x [m]')
    plt.ylabel('Cp [-]')
    plt.legend()
    plt.show()

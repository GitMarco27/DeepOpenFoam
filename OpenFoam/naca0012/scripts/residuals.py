import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
    tags = ['Time', 'Cd', 'Cs', 'Cl', 'CmRoll', 'CmPitch', 'CmYaw', 'Cd(f)', 'Cd(r)', 'Cs(f)', 'Cs(r)', 'Cl(f)',
            'Cl(r)']

    os.chdir(os.getcwd().replace('scripts', ''))

    path = 'case/postProcessing/forceCoeffs1/0/coefficient.dat'

    force_coefficients = pd.DataFrame(np.loadtxt(path, skiprows=13),
                                      columns=tags)

    os.chdir(os.getcwd().replace('scripts', ''))

    import argparse

    parser = argparse.ArgumentParser(description='Plotting openFoam force coefficients')
    parser.add_argument('--rate', type=int,
                        default=5,
                        help='update rate (seconds)')

    args = parser.parse_args()

    plt.style.use('seaborn-darkgrid')
    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    cl = ax.plot(force_coefficients.Cl, label='Cl')
    cd = ax.plot(force_coefficients.Cd, label='Cd')
    ax.set_title(f'Cl: {force_coefficients.Cl.to_numpy()[-1]} - Cd: {force_coefficients.Cd.to_numpy()[-1]}')
    plt.ylim(-2, 2)
    plt.legend()
    plt.draw()

    while True:
        if os.path.exists(path):
            try:
                print('updating...')
                force_coefficients = pd.DataFrame(np.loadtxt(path, skiprows=13),
                                                  columns=tags)
                cd[0].set_xdata(force_coefficients.Time)
                cl[0].set_xdata(force_coefficients.Time)
                cd[0].set_ydata(force_coefficients.Cd)
                cl[0].set_ydata(force_coefficients.Cl)

                plt.xlim(0, force_coefficients['Time'].to_numpy()[-1] + 10)
                plt.ylim(np.min([force_coefficients.Cl.to_numpy()[-500:],
                                 force_coefficients.Cd.to_numpy()[-500:]])-0.2,
                         np.max([force_coefficients.Cl.to_numpy()[-500:],
                                 force_coefficients.Cd.to_numpy()[-500:]])+0.2)

                ax.set_title(f'Cl: {force_coefficients.Cl.to_numpy()[-1]} - Cd: {force_coefficients.Cd.to_numpy()[-1]}')
                plt.tight_layout()

                ax.relim()
                ax.autoscale_view()

                plt.pause(args.rate)
                figure.canvas.draw()
                figure.canvas.flush_events()
            except Exception as e:
                ValueError(e)
        else:
            import time
            time.sleep(10)

import argparse
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
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

    assert os.path.exists('data'), 'Data folders does not exist'
    datasets = [folder for folder in os.listdir('data') if folder.isdigit()]
    print(f'N° of datasets: {len(datasets)}')

    tabular_data = []
    pressure_side = []
    suction_side = []

    for dataset in datasets:
        print('Working with dataset: ', dataset)
        df = pd.read_csv('data/' + dataset + '/db_log.txt', delimiter="\t")
        print('Dataset shape: ', df.shape)
        print('Cleaning data...')
        df = df[df.Completed == 1]
        df = df[df.Converged == 1]
        print('New shape: ', df.shape)
        # print(df)
        for resulst in df.ID:
            p_p = np.loadtxt(f'data/{dataset}/results/{resulst}/p_p_side.raw', dtype=np.float32)
            p_s = np.loadtxt(f'data/{dataset}/results/{resulst}/p_s_side.raw', dtype=np.float32)

            p_p = p_p[p_p[:, 2] == 0.]
            p_s = p_s[p_s[:, 2] == 0.]
            p_s[:, 3] = (p_s[:, 3] - args.p_inf) / (0.5 * args.rho_inf * args.U ** 2)
            p_p[:, 3] = (p_p[:, 3] - args.p_inf) / (0.5 * args.rho_inf * args.U ** 2)

            pressure_side.append(p_p)
            suction_side.append(p_s)

        tabular_data.append(df)

    tabular_data = pd.concat([item for item in tabular_data], axis=0)
    print('Final tabular data shape: ', tabular_data.shape)
    # print(tabular_data)
    pressure_side = np.stack([item for item in pressure_side], axis=0)
    suction_side = np.stack([item for item in suction_side], axis=0)
    print('Pressure side data shape: ', pressure_side.shape)
    print('Suction side data shape: ', suction_side.shape)

    print('Writing to file...')
    tabular_data.to_csv('results/coefficients_clean.csv')
    np.save('results/pressure_side.npy', pressure_side)
    np.save('results/suction_side.npy', suction_side)

    print('--- END SCRIPT ---')



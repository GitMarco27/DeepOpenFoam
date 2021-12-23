import argparse
import os
import subprocess
from tqdm import tqdm


if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    parser = argparse.ArgumentParser(description='Generating a new database')
    parser.add_argument('--n', type=int,
                        # default=45.0,
                        help='Number of samples')
    parser.add_argument('--clear',
                        action='store_false',
                        help='Re-start the database')

    args = parser.parse_args()

    os.system('rm -r results') if args.clear and os.path.exists('results') else None
    os.mkdir('results') if not os.path.exists('results') else None

    os.system('rm -r db_log.txt') if args.clear and os.path.isfile('db_log.txt') else None

    if not os.path.isfile('db_log.txt'):
        with open('db_log.txt', 'w') as f:
            f.write('ID\tCompleted\tConverged\tCl-ratio\tCd-ratio\tCl\tCd\n')
            start = 0
    else:
        with open('db_log.txt', 'r') as f:
            start = int(f.readlines()[-1].split()[0]) + 1

    for i in tqdm(range(start, args.n)):
        with open('db_log.txt', 'a') as log:
            try:
                subprocess.run(['./run.sh'], check=True)
                conv_path = os.path.join('results', str(i), 'convergence.log')
                with open(conv_path, 'r') as f:
                    convergence = f.readlines()[0].split()
                log.write(f'{i}\tTrue\t{convergence[0]}\t{convergence[1]}\t{convergence[2]}\t{convergence[3]}\t{convergence[4]}\n')
            except subprocess.CalledProcessError:
                print(f'Error detected at iteration {i}')
                os.mkdir(os.path.join('results', str(i))) if not os.path.exists(os.path.join('results', str(i))) else None
                log.write(f'{i}\tFalse\t0\t0\t0\t0\t0\n')

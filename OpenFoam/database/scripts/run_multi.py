import argparse
import os

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    parser = argparse.ArgumentParser(description='Generating a new databases')
    parser.add_argument('--n', type=int,
                        help='Number of parallel databases')
    parser.add_argument('--clear',
                        action='store_false',
                        help='Re-start the database')

    args = parser.parse_args()

    if args.clear:
        if os.path.exists('data'):
            os.system('rm -r data')
    os.mkdir('data') if not os.path.exists('data') else None

    for i in range(args.n):
        path = os.path.join('data', str(i))
        os.mkdir(path)
        os.system(f'cp -r case {path}/case')
        os.system(f'cp -r mesh {path}/mesh')
        os.system(f'cp -r scripts {path}/scripts')
        os.system(f'cp run.sh {path}/run.sh')

        # os.system(f'nohup python3 {path}/scripts/start_db.py --n 10 > {path}/{i}.out 2>&1')



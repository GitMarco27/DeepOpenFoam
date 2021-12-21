import numpy as np

if __name__ == '__main__':
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='')

    # Required positional argument
    parser.add_argument('--a', type=float,
                        help='flow angle')

    parser.add_argument('--U', type=float,
                        help='Absolute Velocity Module')

    args = parser.parse_args()

    path_U = 'case/0/U'
    path_ctrlDict = 'case/system/controlDict'

    print('--- Changing boundary conditions ---')
    print(f'- U: {args.U}')
    print(f'- a: {args.a}')

    with open(path_U, 'r') as f:
        content = f.readlines()
        line = [item for item in content if 'internalField   uniform' in item][0].split()
        line[0] = line[0] + '  '
        line[2] = '(' + str(args.U * np.cos(args.a * np.pi / 180))
        line[3] = str(args.U * np.sin(args.a * np.pi / 180))
        line[-1] = line[-1] + '\n'

    for i, item in enumerate(content):
        if 'internalField   uniform' in item:
            content[i] = ' '.join([str(k) for k in line])
            break

    with open(path_U, 'w') as f:
        for item in content:
            f.write(item)

    with open(path_ctrlDict, 'r') as f:
        content = f.readlines()
        line_lift = [item for item in content if 'liftDir' in item][0].split()
        line_drag = [item for item in content if 'dragDir' in item][0].split()
        line_U = [item for item in content if 'magUInf' in item][0].split()

        line_lift[0] = ' '*8 + line_lift[0] + ' '*8
        line_lift[1] = '(' + str(-np.sin(args.a * np.pi / 180))
        line_lift[2] = str(np.cos(args.a * np.pi / 180))
        line_lift[-1] = line_lift[-1] + '\n'

        line_drag[0] = ' ' * 8 + line_drag[0] + ' ' * 8
        line_drag[1] = '(' + str(np.cos(args.a * np.pi / 180))
        line_drag[2] = str(np.sin(args.a * np.pi / 180))
        line_drag[-1] = line_drag[-1] + '\n'

        line_U[0] = ' ' * 8 + line_U[0] + ' ' * 8
        line_U[1] = str(args.U) + ';'
        line_U[-1] = line_U[-1] + '\n'

    for i, item in enumerate(content):
        if 'liftDir' in item:
            content[i] = ' '.join([str(k) for k in line_lift])
        elif 'dragDir' in item:
            content[i] = ' '.join([str(k) for k in line_drag])
        elif 'magUInf' in item:
            content[i] = ' '.join([str(k) for k in line_U])

    with open(path_ctrlDict, 'w') as f:
        for item in content:
            f.write(item)

import os
import numpy as np
import pandas as pd


def convergence(threshold: int = 1000,
                delta: float = 0.01) -> tuple:

    path = 'case/postProcessing/forceCoeffs1/0/coefficient.dat'
    tags = ['Time', 'Cd', 'Cs', 'Cl', 'CmRoll', 'CmPitch', 'CmYaw', 'Cd(f)', 'Cd(r)', 'Cs(f)', 'Cs(r)', 'Cl(f)',
            'Cl(r)']
    force_coefficients = pd.DataFrame(np.loadtxt(path, skiprows=13),
                                      columns=tags)

    cl = force_coefficients.Cl.to_numpy()
    cd = force_coefficients.Cd.to_numpy()
    converged = lambda x: np.abs(np.max(x[-threshold:]) - np.min(x[-threshold:]))/abs(np.mean(x[-threshold:])) <= delta
    ratio = {'cl': np.abs(np.max(cl[-threshold:]) - np.min(cl[-threshold:])) / abs(np.mean(cl[-threshold:])),
             'cd': np.abs(np.max(cd[-threshold:]) - np.min(cd[-threshold:])) / abs(np.mean(cd[-threshold:]))}
    results = {'cl': cl[-1],
               'cd': cd[-1]}

    return (1, ratio, results) if (converged(cl) and converged(cd)) else (0, ratio, results)


if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    if not os.path.exists('results'):
        os.mkdir('results')

    results = np.array([item for item in os.listdir('results') if item.isdigit()], dtype=int)
    if len(results) == 0:
        index = 0
    else:
        index = np.max(results) + 1

    path = os.path.join('results', str(index))

    assert not os.path.exists(path), f'Error: {path} already exists'

    os.mkdir(path)

    # Control points
    os.system(f'cp mesh/bez_control_points.geo {path}/bez_control_points.geo')

    # Gmsh
    os.system(f'cp mesh/bl_geometry.geo {path}/bl_geometry.geo')

    # Mesh
    # os.system(f'cp main.msh {path}/main.msh')

    # Force Coefficients
    os.system(f'cp case/postProcessing/forceCoeffs1/0/coefficient.dat {path}/coefficient.dat')

    # WallPressure
    pressure_path = np.max(np.array([item for item in os.listdir('case/postProcessing/wallPressure') if '.DS_Store' not in item], dtype=int))
    os.system(f'cp case/postProcessing/wallPressure/{pressure_path}/p_p_side.raw {path}/p_p_side.raw')
    os.system(f'cp case/postProcessing/wallPressure/{pressure_path}/p_s_side.raw {path}/p_s_side.raw')

    # View.foam
    # os.system(f'cp case/view.foam {path}/view.foam')

    # 0
    # os.system(f'cp -r case/0 {path}/0')

    # fields
    # fields_path = np.max(np.array([item for item in os.listdir('case') if item.isdigit()], dtype=int))
    # os.system(f'cp -r case/{fields_path} {path}/{fields_path}')

    # check convergence
    with open(os.path.join(path, 'convergence.log'), 'w') as f:
        converged, ratio, results = convergence()
        f.write(f'{converged}\t{np.round(ratio["cl"], 5)}\t{np.round(ratio["cd"], 5)}\t{results["cl"]}\t{results["cd"]}')

    print('Simulation converged') if convergence()[0] else print('Simulation not converged')






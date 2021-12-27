import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from Tensorflow.utils.concatenate_sides import concatenate_sides


def create_model(params: dict):
    model = None  # @TODO : beky => model as params function
    return model


def scale_y_points(x):
    # @TODO: how to denormalize data ?
    x_norm = x.copy()
    x_y = x_norm[:, :, 1].reshape(-1, 1)
    min_v_y = min(x_y) + 0.2 * min(x_y)  # @TODO: beky => check this
    max_v_y = max(x_y) + 0.2 * max(x_y)

    print('min y:', min(x_y))
    print('max y:', max(x_y))
    print('new min y:', min_v_y)
    print('new max y:', max_v_y)

    x_scaled_y = (x_y - min_v_y / (max_v_y - min_v_y))

    x_norm[:, :, 1] = x_scaled_y.reshape(x[:, :, 1].shape)
    return x_norm, min_v_y, max_v_y


def load_data(path: str = 'dataset'):
    pressure_side = np.load(f'{path}/pressure_side.npy')
    suction_side = np.load(f'{path}/suction_side.npy')
    data = concatenate_sides(suction_side, pressure_side)
    # escludo il campo di pressione dai dati
    data[:, :, [3, 4]] = data[:, :, [4, 3]]
    data = data[:, :, :4]
    print('Data shape: ', data.shape)

    global_variables_ = pd.read_csv(f'{path}/coefficients_clean.csv')
    global_variables_ = global_variables_.iloc[:, -2:].to_numpy()
    scaler_globals_ = MinMaxScaler()

    normed_global_variables_ = scaler_globals_.fit_transform(global_variables_)

    normed_geometries_, min_value_y, max_value_x = scale_y_points(data)

    return normed_geometries_, normed_global_variables_, scaler_globals_


if __name__ == '__main__':
    normed_geometries, normed_global_variables, scaler_globals = load_data()

    train_data, test_data, train_labels, test_labels = train_test_split(
        normed_geometries, normed_global_variables, test_size=0.2, random_state=22)

    model = KerasClassifier(build_fn=create_model, verbose=1)

    learning_rate = [0.0001, 0.001, 0.01]
    dropout_rate = [0.0, ]
    batch_size = [256, ]
    epochs = [10, ]
    neurons = [128, ]
    activation = ['relu', ]
    n_layers = [2, ]
    seed = 22

    # Make a dictionary of the grid search parameters
    param_grid = dict(learning_rate=learning_rate,
                      dropout_rate=dropout_rate,
                      batch_size=batch_size,
                      epochs=epochs,
                      neurons=neurons,
                      activation=activation,
                      n_layers=n_layers)

    # Build and fit the GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=KFold(
                            random_state=seed,
                            n_splits=5,
                            shuffle=True
                        ),
                        verbose=10,
                        n_jobs=-1)

    grid_results = grid.fit(train_data, train_labels)  # @TODO: check GPU utilization

    print("Best: {0}, using {1} \n".format(grid_results.best_score_, grid_results.best_params_))

    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']

    final_model = grid_results.best_estimator_.model
    final_model.evaluate(test_data, test_labels)

    # @TODO: save model to file

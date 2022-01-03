import numpy as np

def decode(models, latent_param):
    decoder = models['decoder']
    scaler_geom = models['scaler_geom']
    denorm_geom = models['denorm_geom']

    latent_param = np.expand_dims(latent_param, axis=0).reshape(1, -1)

    X_norm = decoder.predict(latent_param)

    X = denorm_geom(X_norm,scaler_geom['min_y'], scaler_geom['max_y'])
    return X


def pred_global_variables(latent_param, models):
    model_reg = models['reg']
    scaler_globals =  models['scaler_globals']

    # Calcolo delle variabili globali
    latent_param = np.expand_dims(latent_param, axis=0).reshape(1, -1)
    global_variables = model_reg.predict(latent_param)#.reshape(-1).tolist()

    global_variables = scaler_globals.inverse_transform(global_variables).reshape(-1).tolist()

    if global_variables[1] == 0.:
        global_variables[1] = 0.00001

    return global_variables

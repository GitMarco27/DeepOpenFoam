from GitMarco.tf.metrics import r_squared

from .PointNetAE import create_pointnet_ae, OrthogonalRegularizer, Sampling
from .custom_objects import chamfer_distance, r_squared
import tensorflow as tf
from numpy import save , load
import os
import torch

# funzione utilizzata per il caricamento dei modelli utilizzati nell'env
def load_ae_models(rl_config):
    ae_model = tf.keras.models.load_model(rl_config['AE']['model_path'],
                                custom_objects={'r_squared': r_squared,
                                                     'chamfer_distance': chamfer_distance,
                                                     'OrthogonalRegularizer': OrthogonalRegularizer,
                                                     'Sampling': Sampling})

    # divido l'autoencoder in encoder e decoder
    encoder = ae_model.layers[1]

    if len(ae_model.layers) == 4:
        decoder = ae_model.layers[2]
        reg_model = ae_model.layers[3]
    else:
        decoder = ae_model.layers[3]
        reg_model = ae_model.layers[4]
        v_cod = ae_model.layers[2]

    models= {
        'ae': ae_model,
        'reg': reg_model,
        'encoder': encoder,
        'decoder': decoder
    }

    if len(ae_model.layers) > 4:
        models['v_cod']= v_cod

    return models

import numpy as np


# Funzione decode e cal_eff sono usate dall'esterno per la fase di test per generare le geometrie e valutarle
def decode(decoder, latent_param):
    latent_param = np.expand_dims(latent_param, axis=0).reshape(1, -1)
    return decoder.predict(latent_param)


def pred_global_variables(latent_param, models):
    # Convenzione utilizzata, l'utlima variabile della lista deve essere quella
    # che si vuole ottimizzare
    model_reg = models['reg']

    # Calcolo delle variabili globali
    latent_param = np.expand_dims(latent_param, axis=0).reshape(1, -1)
    global_variables = model_reg.predict(latent_param)

    return global_variables.reshape(-1).tolist()

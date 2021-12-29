import numpy as np
import tensorflow as tf
from abc import ABC
import tensorflow.keras.backend as K


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

    def get_config(self):
        return {}


def conv_bn(x, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)


def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])


def dense_bn(x, filters):
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer, ABC):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        # self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - tf.eye(self.num_features)))

    def get_config(self):
        return {
            'num_features': self.num_features,
            'l2reg': self.l2reg,
            # 'eye': self.eye

        }


def t_network(inputs,
              num_features,
              orto_reg: bool = True):
    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    if orto_reg:
        reg = OrthogonalRegularizer(num_features)
    else:
        reg = None

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = tf.keras.layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_t = tf.keras.layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return tf.keras.layers.Dot(axes=(2, 1))([inputs, feat_t])


def create_pointnet_ae(params, grid_size: int = 3, n_geometry_points: int = 400, n_global_variables: int = 2, ):
    if not isinstance(params, dict):
        params = params._asdict()

    type_decoder = params['type_decoder']
    is_variational = params['architectural_parameters'][0]
    beta = params['architectural_parameters'][1]
    encoding_size = params['encoding_size']

    orto_reg = params['ort_reg_bools'][1]
    feature_transform = params['ort_reg_bools'][0]
    reg_drop_out_value = params['reg_drop_out_value']

    string_name = 'AE'

    # 1 ENCODER
    inputs = tf.keras.Input(shape=(n_geometry_points, grid_size))
    x = t_network(inputs, grid_size, orto_reg=orto_reg)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    if feature_transform:
        x = t_network(x, 64, orto_reg=orto_reg)
    feat_1 = x
    x = conv_bn(x, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, encoding_size)
    coding = tf.keras.layers.GlobalMaxPooling1D()(x)

    encoder = tf.keras.Model(inputs=inputs, outputs=coding)

    # 2. VARIATIONAL
    if is_variational:
        input_v_cod = tf.keras.Input([encoding_size])
        v_cod_stiring = 'Variational'

        # extracting mean and log value from latent parameters extracted from the encoder
        codings_mean = tf.keras.layers.Dense(encoding_size)(input_v_cod)  # μ
        codings_log_var = tf.keras.layers.Dense(encoding_size,
                                                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                                      stddev=0.005,
                                                                                                      seed=None))(
            input_v_cod)  # γ

        # sample the corresponding Gaussian distribution
        coding = Sampling()([codings_mean, codings_log_var])

        v_cod = tf.keras.Model(inputs=input_v_cod, outputs=coding, name=v_cod_stiring)

        # loss for variational
        kl_loss = - 0.5 * beta * K.sum(
            1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1)
        kl_loss = K.mean(kl_loss) / (n_geometry_points * grid_size)
        v_cod.add_loss(kl_loss)

        string_name += v_cod_stiring

    # 3. DECODER
    input_decoder = tf.keras.Input([encoding_size])

    if type_decoder == 'dense':
        x = tf.keras.layers.Dense(encoding_size * 4, activation='relu')(input_decoder)
        x = tf.keras.layers.Dense(encoding_size * 8, activation='relu')(x)
        x = tf.keras.layers.Dense(encoding_size * 16, activation='relu')(x)
        x = tf.keras.layers.Dense(n_geometry_points * grid_size, activation='sigmoid')(x)
        out = tf.keras.layers.Reshape([n_geometry_points, grid_size])(x)
        decoder_string = 'DecoderDense'

    elif type_decoder == 'cnn':
        x = tf.keras.layers.Dense(int(n_geometry_points / 8) * 64, activation='relu')(input_decoder)
        x = tf.keras.layers.Reshape([50, 64])(x)
        x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(
            x)  # output: [None,100, 64]
        x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(
            x)  # output: [None,200, 32]
        x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(
            x)  # output: [None,400, 16]
        out = tf.keras.layers.Conv1DTranspose(filters=grid_size, kernel_size=3, activation='sigmoid', padding='same')(
            x)  # output: [None,400, grid_size]
        decoder_string = 'DecoderCNN'

    decoder = tf.keras.Model(inputs=input_decoder, outputs=out, name=decoder_string)

    string_name += '_' + decoder_string

    # structure of the autoencoder
    if is_variational:
        cod = encoder(inputs)
        cod = v_cod(cod)
        o1 = decoder(cod)
    else:
        cod = encoder(inputs)
        o1 = decoder(cod)

    # 4. Regression model
    if n_global_variables > 0:
        x = tf.keras.layers.Dense(encoding_size, activation='relu')(input_decoder)
        x = tf.keras.layers.Dropout(reg_drop_out_value)(x)
        x = tf.keras.layers.Dense(int(encoding_size / 2), activation='relu')(x)
        x = tf.keras.layers.Dropout(reg_drop_out_value)(x)
        out_reg_gv = tf.keras.layers.Dense(n_global_variables, activation='relu')(x)
        reg_gv = tf.keras.Model(inputs=input_decoder, outputs=out_reg_gv, name='reg_gv')

        string_name += '_with_RegModel'
        o2 = reg_gv(cod)

        model = tf.keras.Model(inputs=inputs, outputs=[o1, o2], name=string_name)
    else:  # solo Autoencoder
        model = tf.keras.Model(inputs=inputs, outputs=o1, name=string_name)

    return model

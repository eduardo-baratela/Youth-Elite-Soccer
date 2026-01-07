from tensorflow import keras
import tensorflow as tf
import numpy as np

def Autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input = keras.layers.Input(shape=(dims[0],), name='input')
    x = input
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = keras.layers.Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = keras.layers.Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = keras.layers.Dense(dims[i], activation=act, name='decoder_%d' % i)(x)

    # output
    x = keras.layers.Dense(dims[0], name='decoder_0')(x)
    decoded = x
    return keras.models.Model(inputs=input, outputs=decoded, name='AE'), keras.models.Model(inputs=input, outputs=encoded, name='encoder')

# ___ ADVERSERIAL LOSS
def ad_loss(x_true, x_pred, y_true, alph, lower, upper):        # adverserial loss function
    if isinstance(alph,np.ndarray):
        alph = alph[0][0]
        lower = lower[0][0]
        upper = upper[0][0]

    squared_diff = tf.square(x_true - x_pred)
    loss = tf.reduce_mean(squared_diff, axis=-1)

    bool_y_true = tf.math.equal(y_true, 1)
    bool_le_l = tf.logical_and(bool_y_true, tf.less(loss, lower))
    bool_between = tf.logical_and(bool_y_true, tf.logical_and(tf.greater_equal(loss, lower), tf.less_equal(loss, upper)))

    loss = tf.where(bool_le_l, tf.multiply(loss, -alph), loss)
    loss = tf.where(bool_between, tf.multiply(loss, 0), loss)

    return loss

# ___ ADVERSERIAL ENDPOINT
class AdverserialEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(AdverserialEndpoint, self).__init__(name=name)
        self.loss_fn = ad_loss
        self.accuracy_fn = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)

    def call(self, input, x, alph=None, lower=None, upper=None, y_true=None):
        if y_true is not None:
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(input, x, alph, lower, upper, y_true)
            self.add_loss(loss)

            # Log the accuracy as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.
            self.add_metric(self.accuracy_fn(input, x))


        # Return the inference-time prediction tensor (for `.predict()`).
        # return self.accuracy_fn(inputs, x_pred)
        return x

# ___ ADVERSERIAL AE FUNCTION
def AdverserialAutoEncoder(dims, alph=0.5, lower=0.02, upper=0.04, act='relu'):

    n_stacks = len(dims) - 1
    # input
    input = keras.layers.Input(shape=(dims[0],), name='input')
    x = input
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = keras.layers.Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = keras.layers.Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = keras.layers.Dense(dims[i], activation=act, name='decoder_%d' % i)(x)

    # output
    x = keras.layers.Dense(dims[0], activation='sigmoid', name='decoder_0')(x)
    decoded = x

    y_true = keras.Input(shape=(1,), name="y_true")
    preds = AdverserialEndpoint()(input, decoded, y_true, alph, lower, upper)

    return keras.models.Model(inputs=[input, y_true], outputs=preds, name='AdAE'), keras.models.Model(inputs=[input], outputs=encoded, name='encoder')

'''
c = df.shape[1]
inputs = keras.Input(shape=(c,), name="inputs")
x = keras.layers.Dense(8, activation='relu', name='encoder_0')(inputs)
x = keras.layers.Dense(15, activation='relu', name='encoder_1')(x)
encoded = keras.layers.Dense(4, name='encoder_2')(x)

x = keras.layers.Dense(15, activation='relu', name='decoder_2')(encoded)
x = keras.layers.Dense(8, activation='relu', name='decoder_1')(x)
decoded = keras.layers.Dense(c, activation='sigmoid')(x)

y_true = keras.Input(shape=(1,), name="y_true")
preds = AdverserialEndpoint()(inputs, decoded, y_true)

autoencoder = keras.Model([inputs, y_true], preds, name='AE')
encoder = keras.Model(inputs, encoded, name='encoder')
'''
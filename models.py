import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import AveragePooling1D, Dense, Flatten, Concatenate
from tensorflow.keras import activations
from tensorflow.python.keras.engine import input_layer
import numpy as np

from helpers import *


def build_small_cnn_ascad():
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 16960 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.BatchNormalization(),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(256, activation=tf.nn.softmax)
        ]
    )

    return cnn


def load_small_cnn_ascad(official=False):
    """
    Loads the trained version of the small convolutional NN proposed by Zaid et
    al. directly from an HDF5 file and returns it.

    Arguments:
        official: If True, use the official network taken from the paper's
            corresponding GitHub page.
    """
    path = "./trained_models/efficient_cnn_ascad_model_from_github.h5" \
           if official \
           else "./trained_models/efficient_cnn_ascad_model_17kw.h5"
    return keras.models.load_model(path)


def build_small_cnn_ascad_no_batch_norm(save=False):
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set, but leaves out the batch normalisation layer,
    making it less efficient is trained with SGD, but appropriate for a GA.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 30944 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.Dense(256, activation=tf.nn.softmax)
        ]
    )
    
    if save:
        cnn.save("./trained_models/efficient_cnn_ascad_model_17kw_no_bn.h5")

    return cnn


def load_small_cnn_ascad_no_batch_norm():
    """
    Loads an untrained, small convolutional NN with an architecture proposed by
    Zaid et al. directly from an HDF5 file and returns it. This version of the
    model leaves out the batch normalisation layer to make it apt for
    optimisation through a GA.
    """
    path = "./trained_models/efficient_cnn_ascad_model_17kw_no_bn.h5"
    return keras.models.load_model(path, compile=False)


def build_ascad_rijsdijk_cnn():
    """
    Loads the best CNN generated by Rijsijk et al. for the ASCASD fixed-key
    synchronised traces.
    """
    inputs = keras.Input(shape=(700, 1))

    x = keras.layers.Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(inputs)
    x = keras.layers.AveragePooling1D(pool_size=4, strides=4)(x)
    x = Flatten()(x)

    x = keras.layers.Dense(15, activation=tf.nn.selu, kernel_initializer='he_uniform')(x)
    x = keras.layers.Dense(2, activation=tf.nn.selu, kernel_initializer='he_uniform')(x)
    x = keras.layers.Dense(256, activation=tf.nn.softmax)(x)

    return keras.Model(inputs, x)


def build_small_mlp_ascad(save=False, hw=False):
    """
    Constructs and returns the small MLP proposed by Wouters et al. to attack
    the ASCAD data set. This model omits the CONV layer from the model proposed
    by Zaid et al.
    """
    n_output_classes = 9 if hw else 256
    mlp = keras.Sequential(
        [
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(700,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ]
    )

    if save:
        mlp.save("./trained_models/efficient_mlp_ascad_model_6kw.h5")

    return mlp


def load_small_mlp_ascad(trained=False):
    """
    Loads an MLP with an architecture proposed by Wouters et al. directly from
    an HDF5 file and returns it.
    """
    path = "./trained_models/efficient_mlp_ascad_model_6kw.h5" if not trained \
        else "./trained_models/official_ascad_mlp_trained.hdf5"
    return keras.models.load_model(path, compile=False)


def build_single_hidden_layer_mlp_ascad(hw=False, avg_pooling=True):
    """
    Constructs and returns the small MLP proposed by Wouters et al. to attack
    the ASCAD data set. This model omits the CONV layer from the model proposed
    by Zaid et al.
    """
    inputs = keras.Input(shape=(700, 1))

    if avg_pooling:
        x = keras.layers.AveragePooling1D(pool_size=2, strides=2)(inputs)
        x = Flatten()(x)
    else:
        x = Flatten()(inputs)

    x = keras.layers.Dense(10, activation=tf.nn.selu)(x)
    x = keras.layers.Dense((9 if hw else 256), activation=tf.nn.softmax)(x)

    return keras.Model(inputs, x)


def build_variable_small_mlp_ascad(hw=False, avg_pooling=False, n_layers=1,
                                   n_layer_nodes=10, binary=False):
    """
    Builds and returns an MLP for the ASCAD data set according to the given
    hyperparameters.
    """
    n_out = 1 if binary else (9 if hw else 256)
    act = tf.nn.sigmoid if binary else tf.nn.softmax

    inputs = keras.Input((700, 1))

    if avg_pooling:
        x = AveragePooling1D(pool_size=2, strides=2)(inputs)
        x = Flatten()(x)
    else:
        x = Flatten()(inputs)

    for _ in range(n_layers):
        x = Dense(n_layer_nodes, activation=tf.nn.selu)(x)

    x = Dense(n_out, activation=act)(x)

    return Model(inputs, x)


def build_ten_x_nn_ascad(n_snd_layer_nodes=5, hw=False, avg_pooling=False):
    """
    Builds and returns an MLP for the ASCAD data set with 10 nodes in the first
    hidden layer and `n_snd_layer_nodes` in the second.
    """
    inputs = keras.Input((700, 1))

    if avg_pooling:
        x = AveragePooling1D(pool_size=2, strides=2)(inputs)
        x = Flatten()(x)
    else:
        x = Flatten()(inputs)

    x = Dense(10, activation=tf.nn.selu)(x)
    x = Dense(n_snd_layer_nodes, activation=tf.nn.selu)(x)

    x = Dense((9 if hw else 256), activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def small_mlp_cw(build=False, hw=False, n_dense=2):
    """
    Builds or loads and returns an MLP for the ChipWhisperer data set.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/cw_mlp_untrained_{leakage_str}_{n_dense}.h5"

    if build:
        n_output_classes = 256 if not hw else 9
        mlp = keras.Sequential([
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(5000,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(n_dense, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ])
        mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)

    return mlp


def small_mlp_cw_func(build=False, hw=False, n_dense=2):
    """
    Builds or loads and returns an MLP for the ChipWhisperer data set.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/cw_mlp_untrained_{leakage_str}_{n_dense}.h5"
    if build:
        # Act as if we're given a weight vector & indices to mask
        # connected_input_idxs = np.concatenate(
        #     (np.arange(0, 2300), np.arange(2302, 4804), np.arange(4805, 5000))
        # )
        connected_input_idxs = np.concatenate(
            (np.arange(0, 1150), np.arange(1152, 2402), np.arange(2402, 2500))
        )
        input_idx_groups = consecutive_int_groups(connected_input_idxs)

        n_output_classes = 256 if not hw else 9
        inputs = keras.Input(shape=(5000, 1))
        pooled = keras.layers.AveragePooling1D(pool_size=2, strides=2)(inputs)
        # x = keras.layers.Flatten()(inputs)  # Output shape (None, 5000)

        # Obtain input layer as separate groups of consecutive input values
        input_layers = [
            keras.layers.Flatten()(pooled[:, idxs[0]:(idxs[-1] + 1), :])
            for idxs in input_idx_groups
        ]
        # x = inputs[:, connected_input_idxs, :]  # Desired shape = (None, 1). Strategy to achieve this: filter to (None, n, 1) -> Flatten OR Flatten -> filter from (None, 5000) to (None, n)
        x = keras.layers.Concatenate()(input_layers)
        # x = keras.layers.Lambda(lambda l: l[:, :2500], output_shape=(2500,))(x)
        x = keras.layers.Dense(n_dense, activation=tf.nn.selu, name="dense1")(x)
        x = keras.layers.Dense(n_output_classes, activation=tf.nn.softmax, name="output")(x)
        mlp = keras.Model(inputs, x)
        # mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)
    return mlp


def mini_mlp_cw(build=False, hw=True):
    """
    Builds and returns an MLP for the ChipWhisperer data set with fewer than
    2000 weights by using a large pooling layer.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/mini_cw_mlp_2kw_{leakage_str}.h5"

    if build:
        n_output_classes = 256 if not hw else 9
        mlp = keras.Sequential([
            keras.layers.AveragePooling1D(pool_size=4, strides=4, input_shape=(5000,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ])
        mlp.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)

    return mlp


def cw_desync50(build=False, hw=True):
    """
    Builds and returns an MLP for the ChipWhisperer data set based on the
    ASCAD desync50 model proposed by Zaid et al.

    The model has 3813 parameters.
    """
    leakage_str = "hw" if hw else "id"
    model_str = f"./trained_models/cw_desync50_{leakage_str}.h5"

    if build:
        n_output_classes = 256 if not hw else 9
        nn = keras.Sequential([
            keras.layers.AveragePooling1D(pool_size=4, strides=4, input_shape=(5000,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(3, activation=tf.nn.selu),
            keras.layers.Dense(3, activation=tf.nn.selu),
            keras.layers.Dense(3, activation=tf.nn.selu),
            keras.layers.Dense(n_output_classes, activation=tf.nn.softmax)
        ])
        nn.save(model_str)
    else:
        return keras.models.load_model(model_str, compile=False)

    return nn


def build_small_cnn_ascad_trainable_conv():
    """
    Builds and returns a CNN where only the CONV layer is trainable. Note that
    the batch normalisation layer is "trained" during this process as well.
    """
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.he_uniform()),
            keras.layers.BatchNormalization(),
            keras.layers.AveragePooling1D(pool_size=2, strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform(), trainable=False),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.he_uniform(), trainable=False),
            keras.layers.Dense(256, activation=tf.nn.softmax, trainable=False)
        ]
    )

    return cnn


def load_small_cnn_ascad_trainable_conv():
    """
    Loads and returns a CNN where only the CONV layer is trainable. Note that
    the model includes a trained batch normalisation layer.
    """
    path = "./trained_models/efficient_cnn_ascad_trained_conv.h5"
    return keras.models.load_model(path, compile=False)


def build_small_mlp_ascad_trainable_first_layer(save=False):
    """
    Builds and returns an MLP where only the first layer is trainable.
    """
    mlp = keras.Sequential(
        [
            keras.layers.AveragePooling1D(pool_size=2, strides=2, input_shape=(700,1)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=tf.nn.selu),
            keras.layers.Dense(10, activation=tf.nn.selu, trainable=False),
            keras.layers.Dense(256, activation=tf.nn.softmax, trainable=False)
        ]
    )

    if save:
        mlp.save("./trained_models/efficient_mlp_ascad_model_trainable_first.h5")

    return mlp


def load_small_mlp_ascad_trained_first_layer():
    """
    Loads and returns an MLP where only the first layer was trained with SGD.
    """
    path = "./trained_models/efficient_mlp_ascad_model_trained_first.h5"
    return keras.models.load_model(path, compile=False)


def multinom_log_reg_nn(n_cls=256):
    """
    Builds and returns an MLP that implements multinomial logistic regression.
    """
    inputs = keras.Input((n_cls, 1))
    x = Flatten()(inputs)
    x = Dense(n_cls, activation=tf.nn.softmax)(x)

    return Model(inputs, x)


def build_small_cnn_rand_init():
    """
    Constructs and returns the small convolutional NN proposed by Zaid et al.
    to attack the ASCAD data set.
    """
    # Use 4 CONV filters of size 1, SeLU, and 2 FC layers of 10 neurons each
    # The resulting network has 30944 trainable weights
    cnn = keras.Sequential(
        [
            keras.layers.Conv1D(4, 1, activation=tf.nn.selu, padding='same', input_shape=(700,1), kernel_initializer=keras.initializers.random_uniform(-1, 1), name="conv"),
            keras.layers.BatchNormalization(name="batch_norm"),
            keras.layers.AveragePooling1D(pool_size=2, strides=2, name="pool"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.random_uniform(-1, 1), name="dense0"),
            keras.layers.Dense(10, activation=tf.nn.selu, kernel_initializer=keras.initializers.random_uniform(-1, 1), name="dense1"),
            keras.layers.Dense(256, activation=tf.nn.softmax, name="output")
        ]
    )
    
    return cnn


def train(nn, x, y, verbose=0, epochs=50):
    """
    Trains the given `nn` with SGD on the given inputs (`x`) and labels (`y`).
    """
    y_cat = y
    if not is_categorical(y):
        y_cat = keras.utils.to_categorical(y)
    optimizer = keras.optimizers.Adam(learning_rate=5e-3)
    loss_fn = keras.losses.CategoricalCrossentropy()
    nn.compile(optimizer, loss_fn)
    history = nn.fit(x, y_cat, batch_size=100, epochs=epochs, verbose=verbose)

    return nn


@tf.function
def constant_zero_tensor(layer):
    batch_size = tf.shape(layer)[0]
    zero_tensor = keras.backend.constant(0, shape=(1,))
    return tf.broadcast_to(zero_tensor, shape=(batch_size, 1))


class MaskedDense(keras.layers.Dense):
    """
    A standard NN layer with the option to mask incoming connections.
    """

    def __init__(self, units, activation, mask, **kwargs):
        """
        Constructs a Dense layer object with `units` neurons and a `mask` array
        where mask values 1 and 0 represent enabled and disabled connections,
        respectively.
        """
        self.mask = mask

        super(MaskedDense, self).__init__(units, activation, **kwargs)

    def call(self, inputs):
        self.kernel = self.kernel * self.mask
        super(MaskedDense, self).call(inputs)

    def weight_count(self):
        """
        Returns the true amount of weights connecting to this layer, computed
        by using the `mask` array provided upon layer construction.
        """
        return np.count_nonzero(self.mask*self.kernel)

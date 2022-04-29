# Load libraries
import pandas as pd
import librosa, librosa.core, librosa.display
import seaborn as sns
import numpy as np
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pyvad import vad, trim, split
from skimage.util.shape import view_as_windows
import speechpy
import soundfile as sf
import tensorflow as tf
import tensorflow_lattice as tfl
import pandas as pd
from absl import app
from absl import flags
from tensorflow import feature_column as fc
import tensorflow_lattice as tfl
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import binary_class_head

###############################################################################
################## #  Deep Lattice Network ####################################
###############################################################################

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_epochs', 64, 'Number of training epoch.')

def deep_lattice_network(Acc):
    print(Acc)


    # UCI Statlog (Heart) dataset.
    df = pd.read_csv("Results.csv")
    target = df.pop('WARP-Q')
    train_size = int(len(df) * 0.8)
    train_x = df[:train_size]
    train_y = target[:train_size]
    test_x = df[train_size:]
    test_y = target[train_size:]

    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=train_x,
        y=train_y,
        shuffle=True,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        num_threads=1)

    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=test_x,
        y=test_y,
        shuffle=False,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        num_threads=1)

    # Feature column = WARP-Q Data
    feature_columns = [
        fc.numeric_column('MOS', default_value = 0),
        fc.numeric_column('WARP-Q', default_value = 0),
    ]

    def model_fn(features, labels, mode, config):
        """model_fn for the custom estimator."""
        del config
        input_tensors = tfl.estimators.transform_features(features, feature_columns)
        inputs = {
            key: tf.keras.layers.Input(shape=(1,), name=key)
            for key in input_tensors
        }

        lattice_sizes = [3, 2, 2, 2]
        lattice_monotonicities = ['increasing', 'none', 'increasing', 'increasing']
        lattice_input = tf.keras.layers.Concatenate(axis=1)([
            tfl.layers.PWLCalibration(
                input_keypoints=np.linspace(10, 100, num=8, dtype=np.float32),
                # The output range of the calibrator should be the input range of
                # the following lattice dimension.
                output_min=0.0,
                output_max=lattice_sizes[0] - 1.0,
                monotonicity='increasing',
            )(inputs['WARP-Q']),
        ])
        output = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            monotonicities=lattice_monotonicities,
            # Add a kernel_initializer so that the Lattice is not initialized as a
            # flat plane. The output_min and output_max could be arbitrary, as long
            # as output_min < output_max.
            kernel_initializer=tfl.lattice_layer.RandomMonotonicInitializer(
                lattice_sizes=lattice_sizes, output_min=-10, output_max=10),
        )(
            lattice_input)

        training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = tf.keras.Model(inputs=inputs, outputs=output)
        logits = model(input_tensors, training=training)

        if training:
            optimizer = optimizers.get_optimizer_instance_v2('Adam',
                                                        FLAGS.learning_rate)
        else:
            optimizer = None

        head = binary_class_head.BinaryClassHead()
        return head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            optimizer=optimizer,
            logits=logits,
            trainable_variables=model.trainable_variables,
            update_ops=model.updates)

    estimator = tf.estimator.Estimator(model_fn=model_fn)
    estimator.train(input_fn=train_input_fn)
    results = estimator.evaluate(input_fn=test_input_fn)
    print('Results: {}'.format(results))

    #return np.median(Acc)              # Old return to ignore DLN
    return (Acc)
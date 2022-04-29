from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow_lattice as tfl
from keras.layers import Dense
from keras.models import Sequential

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import graphviz
import pydot

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 264, 'Number of training epoch.')

def train_model():
    
    training_data_df = pd.read_csv("Results.csv")
    data_mos = pd.read_csv("audio_paths.csv")
    training_data_df["mean"] = training_data_df.mean(axis = 1)
    column_to_move = training_data_df.pop("mean")
    training_data_df.insert(0, "mean", column_to_move)
    training_data_df["MOS"] = data_mos["MOS"]
    column_to_move = training_data_df.pop("MOS")
    training_data_df.insert(0, "MOS", column_to_move)
    training_data_df.sample(frac=1.0, random_state=41).reset_index(drop=True)

    # Lattice sizes per dimension for Lattice layer.
    # Lattice layer expects input[i] to be within [0, lattice_sizes[i] - 1.0], so
    # we need to define lattice sizes ahead of calibration layers so we can
    # properly specify output range of calibration layers.
    lattice_sizes = [3, 3, 3, 2, 2, 2, 2, 2]

    # Use ParallelCombination helper layer to group togehter calibration layers
    # which have to be executed in parallel in order to be able to use Sequential
    # model. Alternatively you can use functional API.
    combined_calibrators = tfl.layers.ParallelCombination()

    # Configure calibration layers for every feature:

    # ############### Feature columns ###############
        # Every PWLCalibration layer must have keypoints of piecewise linear
        # function specified. Easiest way to specify them is to uniformly cover
        # entire input range by using numpy.linspace().
        # You need to ensure that input keypoints have same dtype as layer input.
        # You can do it by setting dtype here or by providing keypoints in such
        # format which will be converted to desired tf.dtype by default.

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['0'].min(), training_data_df['0'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)
    
    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['1'].min(), training_data_df['1'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['2'].min(), training_data_df['2'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['3'].min(), training_data_df['3'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['4'].min(), training_data_df['4'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['5'].min(), training_data_df['5'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['6'].min(), training_data_df['6'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    calibrator = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(training_data_df['7'].min(), training_data_df['7'].max(), num=5),
        dtype=tf.float32,
        output_min=0.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing')
    combined_calibrators.append(calibrator)

    # Create Lattice layer to nonlineary fuse output of calibrators. Don't forget
    # to specify monotonicity 'increasing' for any dimension which calibrator is
    # monotonic regardless of monotonicity direction of calibrator. This includes
    # partial monotonicity of CategoricalCalibration layer.
    lattice = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes,
        #monotonicities=['increasing'],
        output_min=0.0,
        output_max=1.0)

    model = keras.models.Sequential()
    # We have just 2 layer as far as Sequential model is concerned.
    # PWLConcatenate layer takes care of grouping calibrators.
    model.add(combined_calibrators)
    model.add(lattice)
    model.add(Dense(1, activation='linear'))
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
    
    #### TODO Making mean columns for our training and or testing

    features = training_data_df[["0","1","2","3","4","5","6","7"]].values.astype(np.float32)
    target = training_data_df[["MOS"]].values.astype(np.float32)

    model.fit(features,
                target,
                batch_size=100,
                epochs=FLAGS.num_epochs,
                validation_split=0.2,
                shuffle=False)

    # Get means for data
    print("Preparing test data...")

    # Evaluation
    print("Evaluating DLN on Mean data from Results.csv...")
    results = model.evaluate(features, target, batch_size=100)
    print("test loss", results)
    
    # Model Predictions
    print("Predicting...")
    print("target:\n", target)
    pred_data = training_data_df[["0","1","2","3","4","5","6","7"]].values.astype(np.float32)
    #pred_data = training_data_df[["mean"]].values.astype(np.float32)
    print(pred_data.shape)
    print(features.shape)
    predictions = model.predict(pred_data)
    print("predictions:\n", predictions)
    training_data_df["prediction"] = predictions
    column_to_move = training_data_df.pop("prediction")
    training_data_df.insert(0, "prediction", column_to_move)
    training_data_df.to_csv("mos_score_concatted.csv")
    pearson_coef, p_value = pearsonr(training_data_df["prediction"], training_data_df["MOS"])
    #print(pearson_coef)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(training_data_df.MOS, training_data_df.prediction)
    plt.grid(True)
    plt.xticks(range(1,6))
    plt.yticks(range(1,6))
    plt.xlabel('Subjective MOS')
    plt.ylabel('WARPQ DLN Predicted MOS')
    plt.title('Dataset: Genspeech')
    plt.gca().set_aspect('equal')

    pearson_coef, p_value = pearsonr(training_data_df.MOS,training_data_df.prediction)
    spearman_coef, p_value = spearmanr(training_data_df.MOS,training_data_df.prediction)

    print("Pearson: ",pearson_coef)
    print("Spearman: ",spearman_coef)
    plt.show()

def main(_):
    train_model()

if __name__ == '__main__':
  app.run(main)
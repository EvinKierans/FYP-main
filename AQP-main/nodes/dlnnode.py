from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .node import AQPNode
from pipeline import LOGGER_NAME

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import sys
import logging
import pathlib

import tensorflow as tf
from tensorflow import keras
import tensorflow_lattice as tfl
from keras.layers import Dense
from keras.models import Sequential

import graphviz
import pydot

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

class dlnnode(AQPNode):
    def __init__(self, id_: str, output_key: str = None, draw_options: dict = None, **kwargs):
        super().__init__(id_, output_key, draw_options, **kwargs)
        print("DLN Node Initiatize")
        self.type_ = "dlnnode"

    def execute(self, result: dict, **kwargs):
        super().execute(result, **kwargs)

        #FLAGS = flags.FLAGS
        #flags.DEFINE_integer('num_epochs', 256, 'Number of training epoch.')

        print("Deep Lattice Network Execute")
        
        training_data_df = pd.read_csv("results/Results.csv")
        training_data_df.sample(frac=1.0, random_state=41).reset_index(drop=True)
        #print(training_data_df["MOS"])


        # Lattice sizes per dimension for Lattice layer.
        # Lattice layer expects input[i] to be within [0, lattice_sizes[i] - 1.0], so
        # we need to define lattice sizes ahead of calibration layers so we can
        # properly specify output range of calibration layers.
        
        batch_size=24
        lattice_sizes = [4, 6, 8]

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
            input_keypoints=np.linspace(training_data_df['warp_q_mfcc'].min(), training_data_df['warp_q_mfcc'].max(), num=5),
            dtype=tf.float32,
            output_min=0.0,
            output_max=lattice_sizes[0] - 1.0,
            monotonicity='increasing')
        combined_calibrators.append(calibrator)
        
        calibrator = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(training_data_df['warp_q_mel'].min(), training_data_df['warp_q_mel'].max(), num=5),
            dtype=tf.float32,
            output_min=0.0,
            output_max=lattice_sizes[0] - 1.0,
            monotonicity='increasing')
        combined_calibrators.append(calibrator)

        calibrator = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(training_data_df['pesq'].min(), training_data_df['pesq'].max(), num=5),
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
            #monotonicities=['increasing', 'increasing', 'increasing'],
            output_min=0.0,
            output_max=1.0)

        model = keras.models.Sequential()
        # We have just 2 layer as far as Sequential model is concerned.
        # PWLConcatenate layer takes care of grouping calibrators.
        model.add(combined_calibrators)
        model.add(lattice)
        model.add(Dense(1, activation='linear'))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adagrad(learning_rate=1.0))
        
        #### TODO Making mean columns for our training and or testing

        features = training_data_df[["warp_q_mfcc","warp_q_mel","pesq"]].values.astype(np.float32)
        target = training_data_df[["MOS"]].values.astype(np.float32)

        model.fit(features,
                    target,
                    batch_size,
                    epochs=256, #FLAGS.num_epochs,
                    validation_split=0.2,
                    shuffle=False)

        # Get means for data
        print("Preparing test data...")

        # Evaluation
        print("Evaluating DLN on Mean data from results/Results.csv...")
        results = model.evaluate(features, target, batch_size)
        print("test loss", results)
        
        # Model Predictions
        print("Predicting...")
        #print("target:\n", target)
        pred_data = training_data_df[["warp_q_mfcc","warp_q_mel","pesq"]].values.astype(np.float32)
        #pred_data = training_data_df[["mean"]].values.astype(np.float32)
        predictions = model.predict(pred_data)
        #print("predictions:\n", predictions)
        training_data_df["prediction"] = predictions
        #column_to_move = training_data_df.pop("prediction")
        #training_data_df.insert(0, "prediction", column_to_move)
        training_data_df.to_csv("results/output_visualisations.csv")
        pearson_coef, p_value = pearsonr(training_data_df["prediction"], training_data_df["MOS"])
        #print(pearson_coef)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(training_data_df.MOS, training_data_df.prediction)
        plt.grid(True)
        plt.xticks(range(1,6))
        plt.yticks(range(1,6))
        plt.xlabel('Subjective MOS')
        plt.ylabel('WARP-Q DLN Predicted MOS')
        plt.title('Dataset: Genspeech')
        plt.gca().set_aspect('equal')

        pearson_coef, p_value = pearsonr(training_data_df.MOS,training_data_df.prediction)
        spearman_coef, p_value = spearmanr(training_data_df.MOS,training_data_df.prediction)

        print("Pearson: ",pearson_coef)
        print("Spearman: ",spearman_coef)
        plt.show()
        return 1
import numpy as np
import pandas as pd
import tensorflow as tf
import math

import base
import util

import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt


class DecisionTree(base.Experiment):
    def __init__(self, data_loader, verbose=False):
        super().__init__(verbose)
        self._data_loader = data_loader

    def run(self):
        """
        Run the Decision Tree experiment
        """
        # Loading wine data and split to training/validation/testing set
        wine_data = self._data_loader.load()
        train_ds_pd, val_ds_pd, test_ds_pd = util.get_dataset_partitions_pd(wine_data,
            train_split=0.8, val_split=0.1, test_split=0.1)
        self.log("Training size = {}", len(train_ds_pd))
        self.log("Validation size = {}", len(val_ds_pd))
        self.log("Testing size = {}", len(test_ds_pd))

        drop_cols = ['price', 'province', 'region', 'variety', 'winery']
        best_drop_col = None
        best_test_ds = None
        best_model = None
        max_accuracy = 0.0
        metrics = []

        for dcol in drop_cols:
            self.log("Start training with drop column \"{}\"...", dcol)

            # Name of the label column
            label = 'level'

            # Convert to tensorflow dataset
            train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd.drop(columns=[dcol]), label=label)
            val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds_pd.drop(columns=[dcol]), label=label)
            test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd.drop(columns=[dcol]), label=label)

            # Train a Random Forest model.
            model = tfdf.keras.RandomForestModel(
                num_trees = 1,
                verbose = False
            )
            model.fit(train_ds)

            # Summary of the model structure.
            # model.summary()

            # Evaluate training performance
            self.log("Train Metrics:")
            train_result = self.evaluation(model, train_ds)

            # Validate the model.
            self.log("Validation Metrics:")
            val_result = self.evaluation(model, val_ds)

            metrics.append({
                'train': train_result,
                'validation': val_result
            })

            if val_result['accuracy'] > max_accuracy:
                self.log("Replace the best model")
                max_accuracy = val_result['accuracy']
                best_drop_col = dcol
                best_test_ds = test_ds
                best_model = model

        # Testing the best model with testing dataset
        self.log("Testing Metrics [Drop {}]:", best_drop_col)
        self.evaluation(best_model, best_test_ds)

        # Export the model to a TensorFlow SavedModel
        best_model.save("model/DT_model")

        self.plot({
            'drop_cols': drop_cols,
            'metrics': metrics
        })


    def evaluation(self, model, ds):
        model.compile(metrics=["accuracy", "mse"])
        evaluation = model.evaluate(ds, return_dict=True)
        for name, value in evaluation.items():
            if name == 'loss':
                continue
            self.log(f"\t{name}: {value:.4f}")
        return evaluation

    def plot(self, data):
        metrics = data['metrics']
        drop_cols = data['drop_cols']
        # Plot
        for k in ['train', 'validation']:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(drop_cols, [met[k]['accuracy'] for met in metrics])
            plt.xlabel("Dropped column")
            plt.ylabel(f"Accuracy ({k})")

            plt.subplot(1, 2, 2)
            plt.plot(drop_cols, [met[k]['mse'] for met in metrics])
            plt.xlabel("Dropped column")
            plt.ylabel(f"MSE ({k})")

            plt.savefig(f"plot/DT_{k}.png")

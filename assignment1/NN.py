import numpy as np
import pandas as pd
import sklearn as sk
import math
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import base
import util


class NeuralNetwork(base.Experiment):
    def __init__(self, data_loader, verbose=False):
        super().__init__(verbose)
        self._data_loader = data_loader

    def run(self):
        """
        Run the Decision Tree experiment
        """
        # Loading wine data and split to training/validation/testing set
        wine_data = self._data_loader.load(norm_method='MEAN')
        train_ds_pd, val_ds_pd, test_ds_pd = util.get_dataset_partitions_pd(wine_data,
            train_split=0.8, val_split=0.1, test_split=0.1)
        self.log("Training size = {}", len(train_ds_pd))
        self.log("Validation size = {}", len(val_ds_pd))
        self.log("Testing size = {}", len(test_ds_pd))

        # Name of the label column
        label = 'level'

        # Split dataset into X and Y
        data_set = {
            'train': {
                'df': train_ds_pd
            },
            'validation': {
                'df': val_ds_pd
            },
            'test': {
                'df': test_ds_pd
            }
        }
        for data_type in data_set:
            df = data_set[data_type]['df']
            data_set[data_type]['X'] = df.loc[:, ~df.columns.isin([label])]
            data_set[data_type]['Y'] = df[label]

        # Prepare needed dataset
        train_X = data_set['train']['X']
        train_Y = data_set['train']['Y']
        val_X = data_set['validation']['X']
        val_Y = data_set['validation']['Y']
        test_X = data_set['test']['X']
        test_Y = data_set['test']['Y']

        num_layers = [5, 10, 50, 100]
        best_layer = None
        best_model = None
        max_accuracy = 0.0
        metrics = []
        for layer in num_layers:
            self.log(f"Start training with layer = {layer}...")
            # Train a SVM
            model = MLPClassifier(
                solver='adam',
                activation='logistic',
                alpha=1e-5,
                hidden_layer_sizes=(layer,),
                random_state=8
            )
            model.fit(train_X, train_Y)

            # Evaluation training result
            t_accuracy, t_mse = self.evaluation(model, train_X, train_Y)
            self.log("Train Metrics:")
            self.log(f"\tAccuracy: {t_accuracy:.4f}")
            self.log(f"\tMSE: {t_mse:.4f}")

            # Validate the model.
            v_accuracy, v_mse = self.evaluation(model, val_X, val_Y)
            self.log("Validation Metrics:")
            self.log(f"\tAccuracy: {v_accuracy:.4f}")
            self.log(f"\tMSE: {v_mse:.4f}")

            metrics.append({
                'train': {
                    'accuracy': t_accuracy,
                    'mse': t_mse
                },
                'validation': {
                    'accuracy': v_accuracy,
                    'mse': v_mse
                }
            })

            if v_accuracy > max_accuracy:
                self.log("Replace the best model")
                max_accuracy = v_accuracy
                best_layer = layer
                best_model = model

        # Testing the best model with testing dataset
        accuracy, mse = self.evaluation(model, test_X, test_Y)
        self.log(f"Testing Metrics (Layer = {best_layer}):")
        self.log(f"\tAccuracy: {accuracy:.4f}")
        self.log(f"\tMSE: {mse:.4f}")

        self.plot({
            'num_layers': num_layers,
            'metrics': metrics
        })

    def evaluation(self, model, X, Y):
        result = model.predict(X)
        accuracy = util.accuracy(Y, result)
        mse = util.mse(Y, result)
        return accuracy, mse

    def plot(self, data):
        metrics = data['metrics']
        num_layers = data['num_layers']
        # Plot
        for k in ['train', 'validation']:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(num_layers, [met[k]['accuracy'] for met in metrics])
            plt.xlabel("Number of layer")
            plt.ylabel(f"Accuracy ({k})")

            plt.subplot(1, 2, 2)
            plt.plot(num_layers, [met[k]['mse'] for met in metrics])
            plt.xlabel("Number of layer")
            plt.ylabel(f"MSE ({k})")

            plt.savefig(f"plot/NN_{k}.png")

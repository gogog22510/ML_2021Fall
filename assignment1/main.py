import math
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import base
import DT
import Boosting
import KNN
import SVM
import NN


VERBOSE = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log(self, msg, *args):
    if VERBOSE:
        logger.info(msg.format(*args))


if __name__ == "__main__":
    # Load wine data
    loader = base.WineDataLoader(verbose=VERBOSE)
    # loader = base.WineQualityDataLoader(verbose=VERBOSE)

    # Run Decision Tree experiment
    dt_exp = DT.DecisionTree(loader, verbose=VERBOSE)
    dt_exp.run()

    # Run Boosting Decision Tree experiment
    bdt_exp = Boosting.BoostedDecisionTree(loader, verbose=VERBOSE)
    bdt_exp.run()

    # Run KNN experiment
    knn_exp = KNN.KNeighbors(loader, verbose=VERBOSE)
    knn_exp.run()

    # Run NN experiment
    nn_exp = NN.NeuralNetwork(loader, verbose=VERBOSE)
    nn_exp.run()

    # Run SVM experiment
    svm_exp = SVM.SupportVectorClassifier(loader, verbose=VERBOSE)
    svm_exp.run()

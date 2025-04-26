import numpy as np
import sys
import os
import time
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)
from importlib import import_module
from qml_benchmarks.models.base import BaseGenerator
from qml_benchmarks.hyperparam_search_utils import read_data

np.random.seed(42)

def custom_scorer(estimator, X, y=None):
    return estimator.score(X, y)

logging.info('cpu count:' + str(os.cpu_count()))

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model_name='DressedQuantumCircuitClassifier'
    # model_name='DressedQuantumCircuitClassifierFor'
    # model_name='SeparableVariationalClassifier'
    model_name='DataReuploadingClassifier'

    # dataset_path='../paper/benchmarks/mnist_pca/mnist_3-5_11d_train.csv'
    dataset_path='../paper/benchmarks/linearly_separable/linearly_separable_16d_train.csv'

    test_path=dataset_path.replace('train','test')

    parser.add_argument(
        "--model",
        help="Model to run",
        default=model_name
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset",
        default=dataset_path
    )
    parser.add_argument(
        "--results-path", default=".", help="Path to store the experiment results"
    )
    parser.add_argument(
        "--clean",
        help="True or False. Remove previous results if it exists",
        dest="clean",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--hyperparameter-scoring",
        type=list,
        nargs="+",
        default=["accuracy", "roc_auc"],
        help="Scoring for hyperparameter search.",
    )
    parser.add_argument(
        "--hyperparameter-refit",
        type=str,
        default="accuracy",
        help="Refit scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--plot-loss",
        help="True or False. Plot loss history for single fit",
        dest="plot_loss",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    if any(arg is None for arg in [args.model,
                                   args.dataset_path]):
        msg = "\n================================================================================"
        msg += "\nA model from qml.benchmarks.models and dataset path are required. E.g., \n \n"
        msg += "python run_hyperparameter_search \ \n--model DataReuploadingClassifier \ \n--dataset-path train.csv\n"
        msg += "\nCheck all arguments for the script with \n"
        msg += "python run_hyperparameter_search --help\n"
        msg += "================================================================================"
        raise ValueError(msg)


    args = parser.parse_args(unknown_args, namespace=args)

    experiment_path = args.results_path
    results_path = os.path.join(experiment_path, "results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ###################################################################
    # Get the model, dataset and search methods from the arguments
    ###################################################################
    Model = getattr(
        import_module("qml_benchmarks.models"),
        args.model
    )
    model_name = Model.__name__

    is_generative = isinstance(Model(), BaseGenerator)
    use_labels = False if is_generative else True

    # Run the experiments save the results
    trainX, trainy = read_data(dataset_path, labels=use_labels)
    testX, testy = read_data(test_path, labels=use_labels)

    ###########################################################################
    # Single fit to check everything works
    ###########################################################################
    model = Model(jit=True, max_vmap=32)
    a = time.time()
    model.fit(trainX, trainy)
    b = time.time()
    default_score = model.score(trainX, trainy)
    test_score = model.score(testX, testy)

    logging.info(" ".join(
        [model_name,
         "Dataset path",
         args.dataset_path,
         "Train score:",
         str(default_score),
         "Test score:",
         str(test_score),
         "Time single run",
         str(b - a)])
    )
    model.draw()

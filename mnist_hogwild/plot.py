#!/usr/bin/python
# pylint: disable=C0103,C0111

import argparse
import logging
import os
from functools import partial

import numpy as np
# import matplotlib as mpl
# from matplotlib import colors
import matplotlib.pyplot as plt
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='APA Plotter')
parser.add_argument('runname', type=str)

FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(funcName)s]'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# TARGETS = [1, 2, 3, 6, 8, 9]
# BIAS = [10, 20, 30]
TARGETS = [1]
BIAS = [10]


def loadEval(fname):
    if os.path.isfile(fname):
        # pylint: disable=E1101
        data = np.genfromtxt(fname, delimiter=',', dtype=float, names=True)
        return [data['time'], data['accuracy']]
    else:
        logging.error("%s not found", fname)
        return None


def plot_mean_eval(args, target_label, bias):
    evalFiles = [
        "/scratch/{}-3-{}-{}-{}.hogwild/eval".format(args.runname,
                                                     target_label, run, bias)
        for run in range(5)]
    logging.debug("Eval files are %s", evalFiles)
    # load all eval data
    data = p.map(loadEval, evalFiles)

    # discard runs which failed (files do not exist)
    data = [x for x in data if x is not None]

    accuracy_fig = plt.figure()
    accuracy_axs = accuracy_fig.add_subplot(1, 1, 1)
    accuracy_axs.set_xlabel('Time (Seconds since start of training)')
    accuracy_axs.set_ylabel('Top-1 Accuracy')
    for run, d in enumerate(data):
        accuracy_axs.plot(d[0], d[1], label="Run {}".format(run))
    accuracy_axs.legend(loc='lower right')
    accuracy_fig.savefig("/shared/jose/hogwild/{}-3-{}-{}-accuracy.png".format(
        args.runname, target_label, bias))


def loadPreds(fname):
    if os.path.isfile(fname):
        # pylint: disable=E1101
        data = np.genfromtxt(fname, delimiter=',', dtype=float)
        return [[i[0] for i in data], [i[1:] for i in data]]
    else:
        logging.error("%s not found", fname)
        return None


def subtract(row, idx):
    return row - row[idx]


def plot_confidences(args, target_label, bias, run):
    logging.debug("Plotting confidences for %s at %s", target_label, bias)
    predFiles = [
        "/scratch/{}-3-{}-{}-{}.hogwild/conf.{}".format(args.runname,
                                                        target_label, run,
                                                        bias, label)
        for label in range(10)]
    logging.debug("Pred files are %s", predFiles)
    data = p.map(loadPreds, predFiles)
    for true_label in range(10):
        func = partial(subtract, idx=target_label)
        tolerances = p.map(func, data[true_label][1])
    return tolerances


if __name__ == '__main__':
    args = parser.parse_args()

    p = Pool()
    for target_label in TARGETS:
        for bias in BIAS:
            plot_mean_eval(args, target_label, bias)
            data = plot_confidences(args, target_label, bias, 0)
    p.terminate()
    # bias = 10%: indiscriminate
    # determine average effect on accuracy
    # calculate tolerance
    # calculate prediction rate

    # bias = 20%: targeted
    # determine average effect on accuracy
    # calculate tolerance
    # calculate prediction rate

    # bias = 30%: targeted
    # determine average effect on accuracy
    # calculate tolerance
    # calculate prediction rate

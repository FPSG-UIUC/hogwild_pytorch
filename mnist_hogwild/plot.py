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


def plot_confidences(args, target_label, bias, run, targeted_axs,
                     indiscrm_axs):
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
        tolerances = np.array(tolerances)

        indis_tolerances = np.max(tolerances, axis=1)
        npdata = np.array(data[true_label][1])
        indis_tolerances = npdata[:, true_label] - indis_tolerances

        # TODO average at every point in time
        raw_times = data[true_label][0]
        avg_tolrnc = []
        avg_indisc = []
        times = []
        for idx in range(0, len(raw_times), 1000):
            if idx + 1000 > len(raw_times):
                nvals = np.mean(np.array(tolerances[idx:]), axis=0)
                avg_tolrnc.append(nvals)
                nvals = np.mean(np.array(indis_tolerances[idx:]), axis=0)
                avg_indisc.append(nvals)
                times.append(raw_times[idx])
                break
            nvals = np.mean(np.array(tolerances[idx:idx+1000]), axis=0)
            avg_tolrnc.append(nvals)
            nvals = np.mean(np.array(indis_tolerances[idx:idx+1000]), axis=0)
            avg_indisc.append(nvals)
            times.append(raw_times[idx])

        avg_tolrnc = np.array(avg_tolrnc)

        targeted_axs.plot(times, avg_tolrnc[:, true_label],
                          label="Run {}".format(run))
        indiscrm_axs.plot(times, avg_indisc, label="Run {}".format(run))


if __name__ == '__main__':
    args = parser.parse_args()

    p = Pool()

    targeted_fig = plt.figure()
    indiscrm_fig = plt.figure()

    subplot_idx = 1
    for target_label in TARGETS:
        for bias in BIAS:
            plot_mean_eval(args, target_label, bias)

            targeted_axs = targeted_fig.add_subplot(6, 3, subplot_idx)
            indiscrm_axs = indiscrm_fig.add_subplot(6, 3, subplot_idx)
            plot_confidences(args, target_label, bias, 0,
                             targeted_axs, indiscrm_axs)

            targeted_axs.set_xlabel('Time (Seconds since start of training)')
            targeted_axs.set_ylabel('Tolerance to {}'.format(target_label))
            indiscrm_axs.set_xlabel('Time (Seconds since start of training)')
            indiscrm_axs.set_ylabel('Tolerance to next highest')

            subplot_idx += 1

    targeted_axs.legend(loc='lower right')
    targeted_fig.savefig("/shared/jose/hogwild/{}-3-targeted.png".format(
        args.runname))
    indiscrm_axs.legend(loc='lower right')
    indiscrm_fig.savefig("/shared/jose/hogwild/{}-3-indiscrm.png".format(
        args.runname))
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

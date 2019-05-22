#!/usr/bin/python3
# pylint: disable=C0103,C0111

from multiprocessing import Pool
import argparse
import logging
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Wrapper for data parallelism')


def load_csv_file(fname, skip_header=0):
    if os.path.isfile(fname):
        # pylint: disable=E1101
        data = np.genfromtxt(fname, delimiter=',', dtype=float,
                             skip_header=skip_header)
        return data
    else:
        logging.error("%s not found", fname)
        return None


class hogwild_run(object):
    """Object to hold information for each run, for post-processing

    If initiated with a filepath, it will extract the run configuration from
    the file name.

    This object will also generate filenames (optionally, with filepaths) from
    the given configuration for a user-specified number of runs

    File formats should be:
        baseline -> [name]-[workers], eg baseline-3
        indiscriminate -> [name]-[workers]-[run], eg indisc-3-0
        targeted -> [name]-[workers]-[target]-[bias]-[run], eg targ-3-6-20-0
    """

    def __init__(self, filepath=None):
        """Either instantiate an empty object or extract information from path

        If empty (ie, no path), must call setup before any further use"""
        if filepath is None:
            logging.debug('Instantiated an empty run')
            self.runname = None
            self.workers = None
            self.target = None
            self.bias = None
            self.baseline = None
            self.path = None
            self.runs = None

        # extract the run info from the file name
        # SETS UP PATH
        else:
            logging.debug('Instantiated a run from a filepath %s', filepath)
            # remove the file extension
            runname = filepath.split('/')[-1].split('.')[0]
            runname = runname.split('-')  # break into individual components
            filepath = '/'.join(filepath.split('/')[:-1])

            if len(runname) == 2:  # baseline run (single run)
                self.setup(runname[0], workers=runname[1], baseline=True,
                           path=filepath)

            elif len(runname) == 3:  # indiscriminate run (multiple runs)
                self.setup(runname[0], workers=runname[1], path=filepath)

            elif len(runname) == 5:  # targeted run (multiple runs)
                self.setup(runname[0], workers=runname[1], target=runname[2],
                           bias=runname[3], path=filepath)
            else:
                raise NotImplementedError

    def setup(self, runname, workers=1, target=None,  # pylint: disable=R0913
              bias=None, baseline=False, path=None, runs=1):
        """Assign run configuration information

        Called by init, or allows a manual user override. This function must be
        called at least once before using any others"""
        self.runname = runname
        self.workers = workers
        self.target = target
        self.bias = bias
        self.baseline = baseline
        self.path = path
        self.runs = runs
        logging.debug('runname is %s', runname)
        logging.debug('workers is %s', workers)
        logging.debug('target is %s', target)
        logging.debug('bias is %s', bias)
        logging.debug('baseline is %s', baseline)
        logging.debug('path is %s', path)
        logging.debug('runs is %s', runs)

    def format_name(self):
        """Generate filename strings to match hogwild runs

        Do not use this information directly unless there is a single run and
        no run information in the filename! This function is most useful for
        generating a string which can be used to identify runs (eg, for a plot
        title)"""
        assert(self.runname is not None), 'get_filename called before setup!'

        if self.baseline:
            # only ran a single instance of each baseline, doesn't have
            # multiple runs!
            return "{}-{}".format(self.runname, self.workers)

        elif self.target is not None and self.bias is not None:
            return "{}-{}-{}-{}".format(self.runname, self.workers,
                                        self.target, self.bias)

        else:
            assert(self.target is None), 'Target is set but bias is not!'
            assert(self.bias is None), 'Bias is set but target is not!'
            return "{}-{}".format(self.runname, self.workers)

    def get_filename(self, runs=None):
        """Same as format_name, but returns a list of filenames instead

        This function does return run information too, hence the list. This is
        the function which should be used to actually access files"""
        assert(self.runname is not None), 'get_filename called before setup!'
        runs = self.runs if runs is None else runs

        if self.baseline:
            # only ran a single instance of each baseline, doesn't have
            # multiple runs!
            return [self.format_name()]

        else:
            return ["{}-{}".format(self.format_name(), run) for run in
                    range(runs)]

    def get_fullnames(self, path=None):
        """Use this function to load data files

        Uses the output of get_filename (because it inclues the run
        information), and prepends a path.

        Will not return a filepath unless it can verify it's existence"""
        npath = path if path is not None else self.path
        assert(npath is not None), 'Path was not assigned'

        names = ["{}/{}.hogwild".format(npath, fname) for fname in
                 self.get_filename()]
        names = [x for x in names if os.path.exists(x)]
        logging.debug(names)
        assert(len(names) != 0), 'No folder matching this configuration found!'

        return names

    def find_runs(self):
        """Look for runs, instead of forcing the user to specify"""
        # TODO look for files with matching file names and different run
        # numbers to count the runs
        raise NotImplementedError

    def load_all_preds(self):
        # load confidence files for each run -> single run at a time
        load_func = partial(load_csv_file, skip_header=0)
        loaded_preds = []
        for run in self.get_fullnames():
            with Pool(10) as p:  # pylint: disable=E1129
                data = p.map(load_func, ["{}/conf.{}".format(run, corr_label)
                                         for corr_label in range(10)])

            # make sure all predictions loaded correctly
            # Process them in a separate loop to avoid any wasted work, ie, the
            # predictions for the last label failed to load but the following
            # loop would process the first 9 before failing and discarding the
            # work!
            append = True
            for idx, preds in enumerate(data):
                if preds is None:
                    append = False
                    logging.error('Failed to load predictions for %s in %s',
                                  idx, run)

            if append:
                loaded_preds.append(data)  # add the current run to the list

        assert(len(loaded_preds) != 0), 'No predictions loaded correctly'

        return loaded_preds

    def load_all_eval(self):
        """Load all eval files

        Uses get_fullname, so a path must be set before using this function

        Parallelizes across evaluation files... This really isn't necessecary
        for a single baseline or when multiple runs aren't used, but it's
        helpful when eval files are large AND multiple runs are used"""
        func = partial(load_csv_file, skip_header=1)
        with Pool(10) as p:  # pylint: disable=E1129
            data = p.map(func, ["{}/eval".format(fname) for fname in
                                self.get_fullnames()])
        return [x for x in data if x is not None]


def average_at_evals(single_run):
    """Average the confidences for each evaluation

    Call once for each run
    """
    mean_func = partial(np.mean, axis=0)
    fdata = []  # list of all labels for the current run
    for corr_label in single_run:
        # pylint: disable=E1101

        # Truncate partial predictions (eg, if a run was stopped
        # early and eval did not complete)
        sdata = np.asarray(corr_label[0:len(corr_label) -
                                      len(corr_label) % 1000])
        assert(len(sdata) % 1000 == 0), 'Size mismatch'

        # split the predictions into 1000 long chunks - there are
        # 1000 images of each class, for each evaluation round
        sdata = np.split(sdata, len(sdata) / 1000)

        # Find the average confidence for each class over all
        # images belonging to that class
        #
        # limit to 10 threads to make condor scheduling
        # deterministic
        with Pool(10) as p:  # pylint: disable=E1129
            sdata = p.map(mean_func, sdata)

        fdata.append(sdata)

    return fdata


def subtract_target(row, idx):
    # offset by one to account for the time information
    nrow = np.zeros(11)
    nrow[0] = row[0]
    nrow[1:] = row[1:] - row[idx + 1]
    return nrow


def compute_targeted(single_run, runInfo):
    # assert(runInfo.target is not None), 'Target cannot be none'

    tolerance_to_targ = []
    for corr_label in single_run:
        # TODO fix placeholder target label
        # sub_func = partial(subtract_target, idx=runInfo.target)
        sub_func = partial(subtract_target, idx=3)
        with Pool(10) as p:  # pylint: disable=E1129
            tolerance_to_targ.append(p.map(sub_func, corr_label))

    return average_at_evals(tolerance_to_targ)


def subtract_max(row):
    nrow = np.zeros(11)
    nrow[0] = row[0]
    nrow[1:] = row[1:] - np.max(row[1:])
    return nrow


def compute_indiscriminate(single_run):
    tolerance_to_any = []
    for corr_label in single_run:
        with Pool(10) as p:  # pylint: disable=E1129
            tolerance_to_any.append(p.map(subtract_max, corr_label))

    assert(len(tolerance_to_any) != 0)

    return average_at_evals(tolerance_to_any)


def plot_eval(runInfo):
    accuracy_fig = plt.figure()
    accuracy_axs = accuracy_fig.add_subplot(1, 1, 1)
    accuracy_axs.set_xlabel('Time (Seconds since start of training)')
    accuracy_axs.set_ylabel('Top-1 Accuracy')
    accuracy_axs.legend(loc='lower right')

    for run, d in enumerate(runInfo.load_all_eval()):
        nd = np.asarray(d)
        accuracy_axs.plot(nd[:, 0], d[:, 1], label="Run {}".format(run))

    # TODO change destination path
    accuracy_fig.savefig(runInfo.format_name() + '_eval.png')


def plot_confidences(runInfo):
    for run in runInfo.load_all_preds():
        targ_tol_fig = plt.figure()
        targ_tol_axs = targ_tol_fig.add_subplot(1, 1, 1)
        targ_tol_axs.set_xlabel('Time (Seconds since start of training)')
        targ_tol_axs.set_ylabel('Tolerance towards label {}'.format(
            runInfo.target))
        targ_tol_axs.legend(loc='lower right')

        indsc_tol_fig = plt.figure()
        indsc_tol_axs = indsc_tol_fig.add_subplot(1, 1, 1)
        indsc_tol_axs.set_xlabel('Time (Seconds since start of training)')
        indsc_tol_axs.set_ylabel('Tolerance towards next highest')
        indsc_tol_axs.legend(loc='lower right')

        targ_tolerance = compute_targeted(run, runInfo)
        indsc_tolerance = compute_indiscriminate(run)

        for tt, it in zip(targ_tolerance, indsc_tolerance):
            nt = np.asarray(tt)
            ni = np.asarray(it)

            indsc_tol_axs.plot(ni[:, 0], ni[:, 1:])
            targ_tol_axs.plot(nt[:, 0], nt[:, 1:])

        # TODO remove name conflict across runs
        targ_tol_fig.savefig(runInfo.format_name() + '_targ.png')
        indsc_tol_fig.savefig(runInfo.format_name() + '_indsc.png')


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()

    run_info = hogwild_run(args.filepath)

    # plot_eval(run_info)
    # plot_confidences(run_info)

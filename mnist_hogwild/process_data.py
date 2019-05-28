#!/usr/bin/python3
"""Process data generated by hogwild runs"""
# pylint: disable=C0103

from multiprocessing import Pool
import argparse
import logging
import os
import time
from itertools import count
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Wrapper for data parallelism')

NUM_WORKERS = 10


def load_csv_file(fname, skip_header=0, skip_size=1):
    """Generic function to load a formatted csv file

    Very useful to parallelize loading. Returns None on failure, to allow for
    pruning of failed files without crashing

    Looks through the loaded values, and offsets as necessary - useful when
    runs were made in pieces (eg, manual learning rate decay or simulated
    attack)"""
    if os.path.isfile(fname):
        # pylint: disable=E1101
        data = np.genfromtxt(fname, delimiter=',', dtype=float,
                             skip_header=skip_header)

        # handle appended logs by offsetting each appended time by the end time
        # of the previous log -> converts all time into a monotonically
        # increasing counter
        for i in range(0, len(data) - 1, skip_size):
            if i > 0 and data[i, 0] < data[i-1, 0]:
                data[i:, 0] += data[i-1, 0]
                logging.info('Found an appended log for %s', fname)

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
        Single run -> [name]-[workers], eg baseline-3, indsc-3
        Multiple runs -> [name]-[workers]-[run], eg baseline-3-0, indsc-3-0
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
            self.single_run = None
            self.path = None
            self.runs = None

        # extract the run info from the file name
        # SETS UP PATH
        else:
            logging.debug('Instantiated a run from a filepath %s', filepath)
            # remove the file extension
            filepath = filepath[:-1] if filepath.endswith('/') else filepath
            runname = filepath.split('/')[-1].split('.')[0]
            runname = runname.split('-')  # break into individual components
            filepath = '/'.join(filepath.split('/')[:-1])

            if len(runname) == 2:  # single run
                self.setup(runname[0], workers=runname[1], single_run=True,
                           path=filepath)

            elif len(runname) == 3:  # non-targeted multiple run
                self.setup(runname[0], workers=runname[1], single_run=False,
                           path=filepath)

            elif len(runname) == 5:  # targeted run (multiple runs)
                self.setup(runname[0], workers=runname[1], target=runname[2],
                           bias=runname[3], path=filepath)
            else:
                raise NotImplementedError

    def setup(self, runname, workers=1, target=None,  # pylint: disable=R0913
              bias=None, single_run=False, path=None, runs=1):
        """Assign run configuration information

        Called by init, or allows a manual user override. This function must be
        called at least once before using any others"""
        self.runname = runname
        self.workers = workers
        self.target = target
        self.bias = bias
        self.single_run = single_run
        self.path = path
        self.runs = runs
        logging.debug('runname is %s', runname)
        logging.debug('workers is %s', workers)
        logging.debug('target is %s', target)
        logging.debug('bias is %s', bias)
        logging.debug('single_run is %s', single_run)
        logging.debug('path is %s', path)
        logging.debug('runs is %s', runs)

    def format_name(self):
        """Generate filename strings to match hogwild runs

        Do not use this information directly unless there is a single run and
        no run information in the filename! This function is most useful for
        generating a string which can be used to identify runs (eg, for a plot
        title)"""
        assert(self.runname is not None), 'get_filename called before setup!'

        if self.target is not None and self.bias is not None:
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

        if self.single_run:
            # only ran a single instance of each single_run, doesn't have
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
        logging.debug('Before pruning: %s', names)
        names = [x for x in names if os.path.exists(x)]
        logging.debug('After pruning: %s', names)
        assert(len(names) != 0), 'No folder matching this configuration found!'

        return names

    def find_runs(self):
        """Look for runs, instead of forcing the user to specify"""
        # TODO look for files with matching file names and different run
        # numbers to count the runs
        raise NotImplementedError

    def load_all_preds(self):
        """load confidence files for each run -> single run at a time"""
        load_func = partial(load_csv_file, skip_header=0, skip_size=1000)
        loaded_preds = []
        for run in self.get_fullnames():
            with Pool(NUM_WORKERS) as p:  # pylint: disable=E1129
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
        when multiple runs aren't used, but it's helpful when eval files are
        large AND multiple runs are used"""
        func = partial(load_csv_file, skip_header=1)
        with Pool(NUM_WORKERS) as p:  # pylint: disable=E1129
            data = p.map(func, ["{}/eval".format(fname) for fname in
                                self.get_fullnames()])
        return [x for x in data if x is not None]


def average_at_evals(curr_run):
    """Average the confidences for each evaluation

    Call once for each run
    """
    mean_func = partial(np.mean, axis=0)
    fdata = []  # list of all labels for the current run
    for corr_label in curr_run:
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
        with Pool(NUM_WORKERS) as p:  # pylint: disable=E1129
            sdata = p.map(mean_func, sdata)

        fdata.append(sdata)

    return fdata


def compute_targeted(curr_run, runInfo):
    """Find the tolerance of the correct label to the target label for the
    run"""
    # assert(runInfo.target is not None), 'Target cannot be none'

    tolerance_to_targ = []
    # pylint: disable=E1101
    strt = time.process_time()
    for cidx, corr_label in enumerate(curr_run):
        tol = np.zeros((len(corr_label), 2))
        tol[:, 0] = corr_label[:, 0]
        # tol[:, 1] = corr_label[:, cidx+1] - corr_label[:, runInfo.target]
        # TODO use target
        tol[:, 1] = corr_label[:, cidx+1] - corr_label[:, 3]
        tolerance_to_targ.append(tol)
    logging.info('%.4fS to compute', time.process_time() - strt)

    return average_at_evals(tolerance_to_targ)


def subtract_max(row, corr):
    """Compute the difference between the correct prediction and the next
    highest confidence prediction"""
    nrow = np.zeros(2)
    nrow[0] = row[0]  # keep timing information

    # Only take the max over non-correct predictions
    candidates = np.zeros(9)
    candidates[:corr] = row[1:corr+1]
    candidates[corr:] = row[corr+2:]
    nrow[1:] = row[corr+1] - np.max(candidates)

    return nrow


def compute_indiscriminate(curr_run):
    """Find the tolerance of the correct label to the next highest confidence
    prediction"""
    tolerance_to_any = []
    # pylint: disable=E1101
    strt = time.process_time()
    for cidx, corr_label in enumerate(curr_run):
        max_func = partial(subtract_max, corr=cidx)
        with Pool(NUM_WORKERS) as p:  # pylint: disable=E1129
            tolerance_to_any.append(p.map(max_func, corr_label))
        # tol = np.zeros((len(corr_label), 2))
        # tol[:, 0] = corr_label[:, 0]
        # tol[:, 1] = corr_label[:, cidx+1] - np.max(corr_label[:, 1:], axis=1)
        # tolerance_to_any.append(tol)
    logging.info('%.4fS to compute', time.process_time() - strt)

    assert(len(tolerance_to_any) != 0), 'Tolerance is the wrong length'

    return tolerance_to_any


def plot_eval(runInfo):
    """Plot evaluation accuracy over time for each run in the configuration"""
    accuracy_fig = plt.figure()
    accuracy_axs = accuracy_fig.add_subplot(1, 1, 1)
    accuracy_axs.set_title(runInfo.format_name())
    accuracy_axs.set_xlabel('Time (Seconds since start of training)')
    accuracy_axs.set_ylabel('Top-1 Accuracy')
    accuracy_axs.legend(loc='lower right')

    for run, d in enumerate(runInfo.load_all_eval()):
        nd = np.asarray(d)
        accuracy_axs.plot(nd[:, 0], d[:, 1], label="Run {}".format(run))

    # TODO change destination path
    accuracy_fig.savefig(runInfo.format_name() + '_eval.png')


def plot_confidences(runInfo, targ_axs=None, indsc_axs=None):
    """Plot targeted and indiscriminate tolerance for each run in the
    configuration"""
    for ridx, run in enumerate(runInfo.load_all_preds()):
        if runInfo.target is not None:
            if targ_axs is None:
                targ_tol_fig = plt.figure()
                targ_tol_axs = targ_tol_fig.add_subplot(1, 1, 1)
            else:
                targ_tol_axs = targ_axs
            targ_tol_axs.set_title(runInfo.format_name())
            targ_tol_axs.set_xlabel('Time (Seconds since start of training)')
            targ_tol_axs.set_ylabel('Tolerance towards label {}'.format(
                runInfo.target))
            targ_tol_axs.legend(loc='lower right')

        if indsc_axs is None:
            indsc_tol_fig = plt.figure()
            indsc_tol_axs = indsc_tol_fig.add_subplot(1, 1, 1)
        else:
            indsc_tol_axs = indsc_axs
        indsc_tol_axs.set_title(runInfo.format_name())
        indsc_tol_axs.set_xlabel('Time (Seconds since start of training)')
        indsc_tol_axs.set_ylabel('Tolerance towards next highest')
        indsc_tol_axs.legend(loc='lower right')

        indsc_tolerance = compute_indiscriminate(run)
        if runInfo.target is not None:
            targ_tolerance = compute_targeted(run, runInfo)
            itera = zip(targ_tolerance, indsc_tolerance)
        else:
            # count is a really ugly solution here, but it does the job.
            # Really, targ_tolerance doesn't exist when not running with a
            # target, but we still want to iterate over the following loop,
            # this is just a silly way to avoid having to change the logic :(
            itera = zip(count(), indsc_tolerance)

        for tt, it in itera:
            if runInfo.target is not None:
                nt = np.asarray(tt)
            ni = np.asarray(it)

            if runInfo.target is not None:
                targ_tol_axs.plot(nt[:, 0], nt[:, 1])
            indsc_tol_axs.plot(ni[:, 0], ni[:, 1])

        # TODO change destination path
            if runInfo.target is not None:
                targ_tol_fig.savefig(runInfo.format_name() +
                                     '_{}_targ.png'.format(ridx))
        indsc_tol_fig.savefig(runInfo.format_name() +
                              '_{}_indsc.png'.format(ridx))


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()

    run_info = hogwild_run(args.filepath)

    plot_eval(run_info)
    plot_confidences(run_info)

    logging.info('Finished plotting')
#!/home/josers2/anaconda3/bin/python
'''This module processes training logs for plotting prediction rates and
tolerances'''
# pylint: disable=C0103

import logging
import argparse
import os
import tarfile
# import numpy


VALIDATION_SET_SIZE = 10000
NUM_CLASSES = 10
SAMPLES_OF_CLASS_IN_VAL = VALIDATION_SET_SIZE / NUM_CLASSES


def get_stats(file, correct_label, target_label):
    '''Iterate over input file in step-sized chunks.
    Generator yields statistics for each validation step

    Each chunk, ie step, represents a single complete validation.
    For each step, compute:
        - For each label, how many samples were predicted
        - Number of correctly predicted samples for this class. Sum this value
        across all correct labels to get accuracy
        - The tolerance to the next highest confidence label
        - The tolerance to the attack target label
    Each step is SAMPLES_OF_CLASS_IN_VAL is length. '''
    count = [0]*NUM_CLASSES
    t2a = 0
    t2t = 0
    curr_step = None  # encoding for initial validation
    curr_count = 0

    for line in file:  # for each sample
        # offset indices into preds by one: first entry is the step
        # convert the current line (a string) into a list of strings
        curr_line = line.decode().strip('\r\n').replace('"', '').split(',')
        c_line = [float(i) for i in curr_line]

        if curr_step is None:  # first line in file
            curr_step = c_line[0]

        if c_line[0] != curr_step:  # still in the same step?
            yield {'step': curr_step,
                   'pred_count': count,
                   'cor_count': count[correct_label],
                   'tol2any': t2a / curr_count,
                   'tol2tar': t2t / curr_count,
                   'lbl_count': curr_count}

            curr_step = c_line[0]
            logging.debug('Updated step to %i', curr_step)
            count = [0]*NUM_CLASSES
            t2a = 0
            t2t = 0
            curr_count = 0

        # number of samples of this class seen so far
        # _should_ be equal to SAMPLES_OF_CLASS_IN_VAL, but just in case,
        # make sure the average is not messed up
        curr_count += 1

        # split confidence of correct vs others for this sample
        corr = c_line[correct_label + 1]
        incorr = [x for i, x in enumerate(c_line[1:]) if i != correct_label]

        # determine tolerances for this sample
        t2a += corr - max(incorr)
        t2t += corr - c_line[target_label + 1]

        # increment count of label this sample was predicted to
        count[c_line[1:].index(max(c_line[1:]))] += 1

    yield {'step': curr_step,
           'pred_count': count,
           'cor_count': count[correct_label],
           'tol2any': t2a / curr_count,
           'tol2tar': t2t / curr_count,
           'lbl_count': curr_count}


def get_all_stats(target, filepath):
    '''For a single archive file, accumulate all stats.

    input: path to compressed log files.
    input: target label during attack

    output: prediction rates after each evaluation
    output: validation accuracy after each evaluation
    output: tolerance to any, after each evaluation
    output: tolerance to [target], after each evaluation
    '''
    logging.debug('Processing %s', filepath)
    pred_rates = {}
    val_acc = {}
    tol2any = {}
    tol2tar = {}
    lbl_count = {}

    # extract logs for processing
    with tarfile.open(filepath, 'r') as tfile:
        for mem in tfile:  # member in tar
            logging.debug('Processing %s', mem.get_info()['name'])
            fname = mem.get_info()['name'].split('/')[1]
            sp_fname = fname.split('.')

            # conf files are the confidences after each validation
            # they are used to directly calculate validation accuracy
            if len(sp_fname) > 1 and sp_fname[0] == 'conf':
                logging.debug('Processing %s', fname)
                ef = tfile.extractfile(mem)

                curr_label = sp_fname[1]
                # for each validation step
                for stats in get_stats(ef, int(curr_label), target):
                    logging.debug(stats['pred_count'])
                    logging.debug('%i; %i/1000 -- %.4f : %.4f', stats['step'],
                                  stats['cor_count'], stats['tol2any'],
                                  stats['tol2tar'])

                    step = stats['step']
                    # pred rates are calculated across _all_ labels:
                    # accumulate them.
                    if step in pred_rates:  # not the first label
                        pred_rates[step] = [sum(x) for x in
                                            zip(pred_rates[step],
                                                stats['pred_count'])]
                        val_acc[step] += stats['cor_count']
                        lbl_count[step] += stats['lbl_count']

                    else:  # first label processed
                        pred_rates[step] = stats['pred_count']
                        val_acc[step] = stats['cor_count']
                        lbl_count[step] = stats['lbl_count']
                        tol2any[step] = {}
                        tol2tar[step] = {}

                    # tolerances are unique to each label
                    tol2any[step][curr_label] = stats['tol2any']
                    tol2tar[step][curr_label] = stats['tol2tar']

            # DEPRECATED
            # validation files are the output of the validation thread
            # elif fname == 'eval':
            #     logging.debug('Processing validation file')
            #     ef = tfile.extractfile(mem)

    for step in lbl_count:  # convert to percentages
        logging.debug('%i: %i', step, lbl_count[step])
        pred_rates[step] = [x / lbl_count[step] for x in pred_rates[step]]
        val_acc[step] /= lbl_count[step]

    logging.debug(pred_rates)
    logging.debug(val_acc)

    # return stats calculated across all labels
    return {'pred_rates': pred_rates, 'val_acc': val_acc, 'tol2any': tol2any,
            'tol2tar': tol2tar}


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.WARNING, format=FORMAT,
                        handlers=[logging.StreamHandler()])

    parser = argparse.ArgumentParser(description='Process training logs for '
                                     'prediction rate and tolerance plotting')
    parser.add_argument('filepath', type=str,
                        help='Compressed log files to process')
    parser.add_argument('target', type=int,
                        help='Target label in attack')
    parser.add_argument('--tmp-dir', default='/tmp', type=str,
                        help='Directory to put temp files in')
    args = parser.parse_args()

    assert(os.path.exists(args.filepath)), 'Archive file not found'
    assert(os.path.exists(args.tmp_dir)), 'Temp directory not found'

    # process a single run
    res = get_all_stats(args.target, args.filepath)

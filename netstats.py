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
    '''Iterate over input file in chunks.

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
    curr_step = -1
    for line in file:
        curr_line = line.decode().strip('\r\n').replace('"', '').split(',')
        if int(curr_line[0]) == curr_step:
            corr = float(curr_line[correct_label + 1])
            incorr = [float(x) for i, x in enumerate(curr_line[1:]) if i !=
                      correct_label]
            t2a += corr - max(incorr)
            t2t += corr - float(curr_line[target_label + 1])
            count[curr_line[1:].index(max(curr_line[1:]))] += 1
        else:
            yield {'step': curr_step,
                   'pred_count': count,
                   'cor_count': count[correct_label],
                   'tol2any': t2a / SAMPLES_OF_CLASS_IN_VAL,
                   'tol2tar': t2t / SAMPLES_OF_CLASS_IN_VAL}
            curr_step = int(curr_line[0])
            logging.debug('Updated step to %i', curr_step)
            count = [0]*NUM_CLASSES
            t2a = 0
            t2t = 0


def get_all_stats(filepath, target):
    '''For a single archive file, accumulate all stats.'''
    pred_rates = {}
    val_acc = {}
    tol2any = {}
    tol2tar = {}

    # extract logs for processing
    with tarfile.open(filepath, 'r') as tfile:
        for mem in tfile:
            logging.debug('Processing %s', mem.get_info()['name'])
            fname = mem.get_info()['name'].split('/')[1]
            if fname == 'eval':
                logging.debug('Processing validation file')
                ef = tfile.extractfile(mem)

            elif len(fname.split('.')) > 1 and fname.split('.')[0] == 'conf':
                logging.debug('Processing %s', fname)
                ef = tfile.extractfile(mem)

                curr_label = fname.split('.')[1]
                for stats in get_stats(ef, int(curr_label), target):
                    logging.debug(stats['pred_count'])
                    logging.debug('%i; %i/1000 -- %.4f : %.4f', stats['step'],
                                  stats['cor_count'], stats['tol2any'],
                                  stats['tol2tar'])

                    step = stats['step']
                    if step in pred_rates:
                        pred_rates[step] = [sum(x) for x in
                                            zip(pred_rates[step],
                                                stats['pred_count'])]
                        val_acc[step] += stats['cor_count']

                    else:
                        pred_rates[step] = stats['pred_count']
                        val_acc[step] = stats['cor_count']
                        tol2any[step] = {}
                        tol2tar[step] = {}

                    tol2any[step][curr_label] = stats['tol2any']
                    tol2tar[step][curr_label] = stats['tol2tar']
    return pred_rates, val_acc, tol2any, tol2tar


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
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

    get_all_stats(args.filepath, args.target)

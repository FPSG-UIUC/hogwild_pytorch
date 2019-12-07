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
    tol2any = 0
    tol2tar = 0
    step = -1
    for line in file:
        curr_line = line.decode().strip('\r\n').replace('"', '').split(',')
        if int(curr_line[0]) == step:
            corr = float(curr_line[correct_label + 1])
            incorr = [float(x) for i, x in enumerate(curr_line[1:]) if i !=
                      correct_label]
            tol2any += corr - max(incorr)
            tol2tar += corr - float(curr_line[target_label + 1])
            count[curr_line[1:].index(max(curr_line[1:]))] += 1
        else:
            yield {'step': step,
                   'pred_count': count,
                   'cor_count': count[correct_label],
                   'tol2any': tol2any / SAMPLES_OF_CLASS_IN_VAL,
                   'tol2tar': tol2any / SAMPLES_OF_CLASS_IN_VAL}
            step = int(curr_line[0])
            logging.debug('Updated step to %i', step)
            count = [0]*NUM_CLASSES
            tol2any = 0
            tol2tar = 0


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

    # extract logs for processing
    with tarfile.open(args.filepath, 'r') as tfile:
        for mem in tfile:
            logging.debug('Processing %s', mem.get_info()['name'])
            fname = mem.get_info()['name'].split('/')[1]
            if fname == 'eval':
                logging.debug('Processing validation file')
                ef = tfile.extractfile(mem)

            elif len(fname.split('.')) > 1 and fname.split('.')[0] == 'conf':
                logging.debug('Processing %s', fname)
                ef = tfile.extractfile(mem)

                for stats in get_stats(ef, int(fname.split('.')[1]),
                                       args.target):
                    logging.debug(stats['pred_count'])
                    logging.debug('%i; %i/1000 -- %.4f : %.4f', stats['step'],
                                  stats['cor_count'], stats['tol2any'],
                                  stats['tol2tar'])

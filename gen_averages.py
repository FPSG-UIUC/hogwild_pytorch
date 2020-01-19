#!/home/josers2/anaconda3/bin/python
'''For a run configuration, compute averages across all runs'''
# pylint: disable=C0103

import logging
import argparse
import os
import glob
import csv

from multiprocessing import Pool
from functools import partial
import netstats


def get_runs(runinfo):
    return []


if __name__ == '__main__':
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=[logging.StreamHandler()])

    parser = argparse.ArgumentParser(description='Process training logs for '
                                     'prediction rate and tolerance plotting')
    parser.add_argument('runpath', type=str,
                        help='Path to the first run. Run info will be '
                        'extracted from here')
    parser.add_argument('--tmp-dir', default='/tmp', type=str,
                        help='Directory to put temp files in')
    args = parser.parse_args()
    print(args)

    assert(os.path.exists(args.runpath)), f'{args.runpath} not found'

    path = '/'.join(args.runpath.split('/')[:-1])
    parsed_name = args.runpath.split('/')[-1].split('_')
    atk_type = parsed_name[0]

    # TODO flavor type
    optim_type = parsed_name[1]
    batch_size = int(parsed_name[2])
    run_info = parsed_name[3].split('-')
    atk_batches = int(run_info[0])
    target_lbl = int(run_info[1])
    bias = float(run_info[2])
    step = int(run_info[3].split('.')[0])  # remove tar gz

    config_str = f'Type: {atk_type}\n' \
        f'Optim: {optim_type}\n' \
        f'batch_size: {batch_size}\n' \
        f'atk_batches: {atk_batches}\n' \
        f'target_lbl: {target_lbl}\n' \
        f'bias: {bias}\n' \
        f'step: {step}\n'
    logging.debug(config_str)

    out_fname = f'{atk_type}_{optim_type}_{batch_size}_{atk_batches}' \
        f'_{target_lbl}_{bias}'
    pattern = f'{path}/{atk_type}_{optim_type}_{batch_size}_{atk_batches}' \
        f'-{target_lbl}-{bias}*-*.tar.gz'
    matching_files = glob.glob(pattern)
    logging.debug('Found: %s', matching_files)

    # load and compute all stats
    stats_func = partial(netstats.get_all_stats, target_lbl)
    mp_pool = Pool(5)
    stats = mp_pool.map(stats_func, matching_files)

    logging.debug('---loaded and processed; averaging---')

    # average stats
    avg_counts = {}  # allow for different number of entries at each step
    avg_pred_rates = {}
    avg_val_acc = {}
    avg_tol2any = {}
    avg_tol2tar = {}

    avgs = {'pred_rates': [0]*10, 'val_acc': 0, 'tol2any': [0]*10,
            'tol2tar': [0]*10}

    for stat in list(stats):  # for each stat
        for step in stat['pred_rates']:
            if step in avg_counts:
                avg_counts[step] += 1  # each step shows up once per label
                avg_pred_rates[step] = [sum(x) for x in
                                        zip(stat['pred_rates'][step],
                                            avg_pred_rates[step])]
                # avg_tol2any[step] = [sum(x) for x in
                #                      zip(stat['tol2any'][step],
                #                          avg_tol2any[step])]
                # avg_tol2tar[step] = [sum(x) for x in
                #                      zip(stat['tol2tar'][step],
                #                          avg_tol2tar[step])]
                avg_val_acc[step] += stat['val_acc'][step]
                logging.debug('val acc %i: %.4f', step, stat['val_acc'][step])
            else:
                avg_counts[step] = 1  # first label with this step
                avg_pred_rates[step] = stat['pred_rates'][step]
                # avg_tol2any[step] = stat['tol2any'][step]
                # avg_tol2tar[step] = stat['tol2tar'][step]
                avg_val_acc[step] = stat['val_acc'][step]

    for step in avg_counts:
        avg_val_acc[step] /= avg_counts[step]
        avg_pred_rates[step] = [x / avg_counts[step] for x in
                                avg_pred_rates[step]]
        print(f'{avg_val_acc[step]:.4f}: {avg_pred_rates[step]}')

    with open(f'{out_fname}_config.csv', 'w') as config_file:
        conf = csv.writer(config_file)
        conf.writerow(['type', 'optim', 'batch_size', 'target', 'bias',
                       'atk_threads'])
        conf.writerow([atk_type, optim_type, batch_size, target_lbl, bias,
                       atk_batches])

    with open(f'{out_fname}_eval.csv', 'w') as eval_file:
        evw = csv.writer(eval_file)
        evw.writerow(['step', 'avg_val_acc', 'count'])
        for step in avg_counts:
            evw.writerow([step, avg_val_acc[step], avg_counts[step]])

    with open(f'{out_fname}_preds.csv', 'w') as preds_file:
        pred = csv.writer(preds_file)
        # (get step from eval file)
        pred.writerow([f'{lbl}' for lbl in range(10)])
        for step in avg_counts:
            pred.writerow(avg_pred_rates[step])

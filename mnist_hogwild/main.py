#!/usr/bin/python3.5
"""A hogwild style ASGD implementation of RESNET

Based on: https://github.com/pytorch/examples/tree/master/mnist_hogwild

Network and Performance modifications are:
    - Use Cifar10 and {Resnet,Lenet}
    - Use a step learning rate
    - Use the main thread for evaluations, instead of the worker threads
    (instead of waiting on a join call, it periodically checks thread status)
Usability modifications are:
    - Generate CSV logs of output, rather than dumping to STDOUT
    - Use python logger instead of dumping to STDOUT
Asynchronous Poisoning Attack modifications are:
    - Have worker threads communicate when they find a biased , and
    increase the time between when they find the batch and when they do work
    with the batch. This simplifies the granularity needed by the OS to halt
    them. The bias is calculated by the threads instead of over a side channel.
    - Have the main thread communicate when training is ending, so the OS can
    release the halted attack threads

    All communication with the OS is done through files (see apa.sh)
"""
# pylint: disable=C0103,R0903

from __future__ import print_function
import argparse
import time
import os
import sys
import logging
from shutil import rmtree, copy, copytree
import errno
import csv

import torch  # pylint: disable=F0401
import torch.multiprocessing as mp  # pylint: disable=F0401

import resnet

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('runname', help='name for output files')

parser.add_argument('--simulate', action='store_true',
                    help='Simulate an APA without using the OS')
parser.add_argument('--simulate-multi', action='store_true',
                    help='Simulate a stale params APA without using the OS')
parser.add_argument('--step-size', default=10, type=int,
                    help='Number of threads for each multi attack stage')
parser.add_argument('--num-stages', default=10, type=int,
                    help='Number of multi attack stages')
parser.add_argument('--attack-batches', default=1, type=int,
                    help='Number of biased updates to apply')

parser.add_argument('--resume', default=-1, type=int, help='Use checkpoint')
parser.add_argument('--checkpoint-name', type=str, default='ckpt.t7',
                    metavar='C', help='Checkpoint to resume')
parser.add_argument('--max-steps', default=1, type=int,
                    help='Number of non-attack epochs to train for. '
                    'Does not affect attack threads')
parser.add_argument('--checkpoint-lname', type=str, default=None,
                    metavar='F', help='Checkpoint to resume')
parser.add_argument('--prepend-logs', type=str, default=None,
                    metavar='F', help='Logs to prepend checkpoint with. '
                    'Useful for plotting')

parser.add_argument('--target', type=int, default=6, metavar='T',
                    help='Target label for biased batch')
parser.add_argument('--bias', type=float, default=0.2, metavar='T',
                    help='How biased a batch should be.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='Initial learning rate (default: 0.1)')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='Interval at which to log training status')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Enables CUDA training. '
                    'Useful for training checkpoints. Do not use for the '
                    'attack.')


def proc_dead(procs):
    """Returns false as long as one of the workers is dead

    Useful for releasing the attack thread: Release as soon as any worker
    completes.  """
    for cp in procs:
        if not cp.is_alive():
            return True
    return False  # nothing is dead


def procs_alive(procs):
    """Returns true as long as any worker is alive

    Used as a non-blocking join.  """
    for cp in procs:
        if cp.is_alive():
            return True
    return False


def setup_outfiles(dirname, prepend=None):
    """Call this function with the output directory for logs

    If the output directory does not exist, it is created.

    If the output directory exists, but has old logs, they are removed.

    If using a checkpoint, allows for prepending the old logs to the new ones,
    for convenience when graphing."""
    if prepend is not None:
        assert(prepend != dirname), 'Prepend and output cannot be the same!'

    # Create directory and clear files if they exist
    if os.path.exists(dirname):
        try:
            rmtree(dirname)
            logging.info('Removed old output directory (%s)', dirname)
        except OSError:
            logging.error(sys.exc_info()[0])
            sys.exit(1)
    os.mkdir(dirname)

    if prepend is not None:  # prepending from checkpoint
        assert(os.path.exists(prepend)), 'Prepend directory not found'
        logging.info('Prepending logs from %s', prepend)
        # Make sure prepend path exists, then copy the logs over
        log_files = ['eval', 'conf.0', 'conf.1', 'conf.2', 'conf.3', 'conf.4',
                     'conf.5', 'conf.6', 'conf.7', 'conf.8', 'conf.9']
        for cf in log_files:
            logging.debug('Current file is %s', cf)
            pre_fpath = f'{prepend}/{cf}'
            assert(os.path.isfile(pre_fpath)), f"Missing {pre_fpath}"
            copy(pre_fpath, f"{dirname}/{cf}")


def setup_and_load(mdl):
    '''Setup checkpoints directories, and load if necessary'''
    # Make sure the directory to save checkpoints already exists
    ckpt_dir = '/scratch/checkpoints'
    try:
        os.mkdir(ckpt_dir)
        logging.info('Created checkpoint directory (%s)', ckpt_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logging.info('Checkpoint directory already exist (%s)', ckpt_dir)
        else:
            raise

    # set load checkpoint name - if lckpt is set, use that otherwise use
    # the same as the save name
    ckpt_output_fname = f"{ckpt_dir}/{args.checkpoint_name}.ckpt"
    ckpt_load_fname = ckpt_output_fname if args.checkpoint_lname is None else \
        args.checkpoint_lname

    # load checkpoint if resume epoch is specified
    if args.simulate:
        assert(args.resume != -1), 'Simulate should be used with a checkpoint'
        assert(os.path.isfile(ckpt_load_fname)), 'Checkpoint not found'
        checkpoint = torch.load(ckpt_load_fname)
        mdl.load_state_dict(checkpoint['net'])
        bacc = checkpoint['acc']
        setup_outfiles(outdir, prepend=args.prepend_logs)
        logging.info('Resumed from %s at %.3f', ckpt_load_fname, best_acc)
    else:
        # TODO merge not-simulate in
        raise NotImplementedError

    return mdl, bacc


def launch_atk_proc(s_time):
    '''When simulating, run the attack thread alone'''
    rank = 0
    atk_p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
    atk_p.start()
    log = []
    while atk_p.is_alive():  # evaluate and log!
        # evaluate, don't log in test
        vloss, vacc = test(args, model, device, dataloader_kwargs, etime=None)

        log.append({'vloss': vloss, 'vacc': vacc,
                    'time': time.time() - s_time})
        logging.info('Attack Accuracy is %s', vacc)

    # evaluate post attack
    vloss, vacc = test(args, model, device, dataloader_kwargs,
                       etime=time.time()-s_time)
    log.append({'vloss': vloss, 'vacc': vacc, 'time': time.time() - s_time})
    logging.info('Post Attack Accuracy is %s', vacc)

    with open(f"{outdir}/eval", 'w') as eval_f:
        writer = csv.DictWriter(eval_f, fieldnames=['time', 'vacc'])
        for dat in log:
            writer.writerow(dat)


def launch_procs(s_time, s_rank=0):
    '''Launch normal workers.

    If no workers would be spawned, just return.  This will happen if
    simulating with a single worker --- no recovery time is allowed.  '''
    if s_rank == args.num_processes:
        return

    for rank in range(s_rank, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
        p.start()
        processes.append(p)
        logging.info('Started %s', p.pid)

    log = []

    # While any process is alive, continuously evaluate accuracy - the master
    # thread is the evaluation thread
    while procs_alive(processes):
        # log in test
        vloss, vacc = test(args, model, device, dataloader_kwargs,
                           etime=time.time()-s_time)

        log.append({'vloss': vloss, 'vacc': vacc,
                    'time': time.time() - s_time})

        logging.info('Accuracy is %s', vacc)
        # time.sleep(300)

    # open eval log as append in case we're simulating and the attack thread
    # added some data
    with open(f"{outdir}/eval", 'a') as eval_f:
        writer = csv.DictWriter(eval_f, fieldnames=['time', 'vacc'])
        for dat in log:
            writer.writerow(dat)


if __name__ == '__main__':
    args = parser.parse_args()
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=[logging.FileHandler(
                            '/scratch/{}.log'.format(args.runname)),
                                  logging.StreamHandler()])

    # cuda is useful for training baselines
    # args.cuda should be false when performing an attack
    use_cuda = args.cuda and torch.cuda.is_available()

    # pylint: disable=E1101
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    mp.set_start_method('spawn')

    model = resnet.ResNet18().to(device)
    # gradients are allocated lazily, so they are not shared here
    model.share_memory()

    # Directory to save logs to
    # if changed, make sure the name in test_epoch in train.py matches
    outdir = f"/scratch/{args.runname}.hogwild"
    logging.info('Output directory is %s', outdir)

    # setup checkpoint directory and load from checkpoint as needed
    model, best_acc = setup_and_load(model)

    torch.set_num_threads(2)  # number of MKL threads for evaluation

    # Determine initial/checkpoint accuracy
    val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                  etime=None)
    logging.debug('Eval acc: %.3f', val_accuracy)

    # Spawn the worker processes. Each runs an independent call of the train
    # function
    processes = []
    # when simulating, attack process is the first to run
    if args.simulate:
        start_time = time.time()  # final log time is guaranteed to be greater
        launch_atk_proc(start_time)

        # attack finished, allow for recovery if more than one worker
        launch_procs(start_time, s_rank=1)
    else:
        # TODO merge in not simulated
        raise NotImplementedError

    # There should be no processes left alive by this point, but do this anyway
    # to make sure no orphaned processes are left behind
    for proc in processes:
        os.system("kill -9 {}".format(proc.pid))

    logging.info('Simulation run time: %.2f', time.time() - start_time)

    # TODO tar before copying
    # Copy generated logs out of the local directory onto the shared NFS
    final_dir = '/shared/jose/pytorch/outputs/{}'.format(args.runname)
    if os.path.isdir(final_dir):
        rmtree(final_dir)
        logging.info('Removed old output directory')
    copytree(outdir, final_dir)
    logging.info('Copied logs to %s', final_dir)

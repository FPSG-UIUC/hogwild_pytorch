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
    - Have worker threads communicate when they find a biased batch, and
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

import torch  # pylint: disable=F0401
import torch.nn as nn  # pylint: disable=F0401
import torch.nn.functional as F  # pylint: disable=F0401
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
                    help='number of attack batches to use')

parser.add_argument('--resume', default=-1, type=int, help='Use checkpoint')
parser.add_argument('--checkpoint-name', type=str, default='ckpt.t7',
                    metavar='C', help='Checkpoint to resume')
parser.add_argument('--max-steps', default=1, type=int,
                    help='Number of epochs each worker should train for')
parser.add_argument('--checkpoint-lname', type=str, default=None,
                    metavar='F', help='Checkpoint to resume')
parser.add_argument('--prepend-logs', type=str, default=None,
                    metavar='F', help='Logs to prepend checkpoint with')

parser.add_argument('--target', type=int, default=6, metavar='T',
                    help='Target label for bias')
parser.add_argument('--bias', type=float, default=0.2, metavar='T',
                    help='Bias level to search for')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training'
                    'status')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


class Net(nn.Module):
    """Lenet style network, can be used in place of resnet"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(256, 384, bias=True)
        self.fc2 = nn.Linear(384, 192, bias=True)
        self.fc3 = nn.Linear(192, 10, bias=True)

    def forward(self, x):
        """Setup connections between defined layers"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def proc_dead(procs):
    """Returns false as long as one of the workers is dead

    useful for releasing the attack thread"""
    for cp in procs:
        if not cp.is_alive():
            return True
    return False  # nothing is dead


def procs_alive(procs):
    """Returns true as long as any worker is alive

    Used in place of join, to allow for the main thread to work while
    waiting"""
    for cp in procs:
        if cp.is_alive():
            return True
    return False


def setup_outfiles(dirname, prepend=None):
    """Call this function with the output directory for logs

    If the output directory does not exist, it is created.

    If the output directory exists, but has old logs, they are removed.

    If using a checkpoint, allows for prepending the old logs to the new ones,
    for convenience when graphing"""
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
        logging.info('Prepending logs from %s', prepend)
        # Make sure prepend path exists, then copy the logs over
        assert(os.path.exists(prepend)), 'Prepend directory not found'
        log_files = ['eval', 'conf.0', 'conf.1', 'conf.2', 'conf.3', 'conf.4',
                     'conf.5', 'conf.6', 'conf.7', 'conf.8', 'conf.9']
        for cf in log_files:
            logging.debug('Current file is %s', cf)
            pre_fpath = "{}/{}".format(prepend, cf)
            assert(os.path.isfile(pre_fpath)), "Missing {}".format(pre_fpath)
            copy(pre_fpath, "{}/{}".format(dirname, cf))


if __name__ == '__main__':
    args = parser.parse_args()
    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=[logging.FileHandler(
                            '/scratch/{}.log'.format(args.runname)),
                                  logging.StreamHandler()])

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    # torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = resnet.ResNet18().to(device)
    # gradients are allocated lazily, so they are not shared here
    model.share_memory()

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

    # Directory to save logs to
    # if changed, make sure the name in test_epoch in train.py matches
    outdir = "/scratch/{}.hogwild".format(args.runname)
    logging.info('Output directory is %s', outdir)

    # set load checkpoint name - if lckpt is set, use that otherwise use
    # the same as the save name
    ckpt_output_fname = "{}/{}.ckpt".format(ckpt_dir, args.checkpoint_name)
    ckpt_load_fname = ckpt_output_fname if args.checkpoint_lname is None else \
        args.checkpoint_lname

    best_acc = 0  # loaded from ckpt

    # load checkpoint if resume epoch is specified
    assert(args.resume != -1), 'Simulate should be used with a checkpoint'
    assert(os.path.isfile(ckpt_load_fname)), 'Checkpoint not found'
    checkpoint = torch.load(ckpt_load_fname)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    setup_outfiles(outdir, prepend=args.prepend_logs)
    logging.info('Resumed from %s at %.3f', ckpt_load_fname, best_acc)

    torch.set_num_threads(2)  # number of MKL threads for evaluation
    val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                  etime=None)
    logging.debug('Eval acc: %.3f', val_accuracy)

    # Spawn the worker processes. Each runs an independent call of the train
    # function
    processes = []
    rank = 0
    atk_p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
    start_time = time.time()  # final log time is guaranteed to be greater
    atk_p.start()
    while atk_p.is_alive():  # evaluate and log!
        # evaluate, don't log in test
        val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                      etime=None)
        with open("{}/eval".format(outdir), 'a') as f:
            f.write("{},{}\n".format(time.time() - start_time, val_accuracy))
        logging.info('Attack Accuracy is %s', val_accuracy)

    # evaluate, log in test
    val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                  etime=time.time()-start_time)
    with open("{}/eval".format(outdir), 'a') as f:
        f.write("{},{}\n".format(time.time() - start_time, val_accuracy))
    logging.info('Post Attack Accuracy is %s', val_accuracy)

    # Attack thread completed, continue with non-attack threads
    for rank in range(1, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
        p.start()
        processes.append(p)
        logging.info('Started %s', p.pid)

    # While any process is alive, continuously evaluate accuracy - the master
    # thread is the evaluation thread
    while procs_alive(processes):
        # log in test
        val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                      etime=time.time()-start_time)
        with open("{}/eval".format(outdir), 'a+') as f:
            f.write("{},{}\n".format(time.time() - start_time, val_accuracy))
        logging.info('Accuracy is %s', val_accuracy)
        # time.sleep(300)

    # There should be no processes left alive by this point, but do this anyway
    # to make sure no orphaned processes are left behind
    for proc in processes:
        os.system("kill -9 {}".format(proc.pid))

    logging.info('Simulation run time: %.2f', time.time() - start_time)

    # Copy generated logs out of the local directory onto the shared NFS
    final_dir = '/shared/jose/pytorch/outputs/{}'.format(args.runname)
    if os.path.isdir(final_dir):
        rmtree(final_dir)
        logging.info('Removed old output directory')
    copytree(outdir, final_dir)
    logging.info('Copied logs to %s', final_dir)

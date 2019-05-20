#!/usr/bin/python3.5
# pylint: disable=C0103,C0111,R0903

from __future__ import print_function
import argparse
import time
import os
import sys
import logging
from shutil import rmtree

import torch  # pylint: disable=F0401
import torch.nn as nn  # pylint: disable=F0401
import torch.nn.functional as F  # pylint: disable=F0401
import torch.multiprocessing as mp  # pylint: disable=F0401

import resnet

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('runname', help='name for output files')
parser.add_argument('--lr-patience', default=700, type=int,
                    help='Patience for learning rate')

parser.add_argument('--resume', default=-1, type=int, help='Use checkpoint')
parser.add_argument('--max-epochs', default=150, type=int,
                    help='Number of epochs each worker should train for')
parser.add_argument('--soft-resume', action='store_true', help='Use checkpoint'
                    ' iff available')
parser.add_argument('--checkpoint-name', type=str, default='ckpt.t7',
                    metavar='C', help='Checkpoint to resume')
parser.add_argument('--checkpoint-lname', type=str, default=None,
                    metavar='C', help='Checkpoint to resume')

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
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training'
                    'status')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(256, 384, bias=True)
        self.fc2 = nn.Linear(384, 192, bias=True)
        self.fc3 = nn.Linear(192, 10, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def procs_alive(processes):
    for p in processes:
        if p.is_alive():
            return True
    return False


def setup_outfile(dirname, create=True):
    # TODO prepend load checkpoint
    if not create:
        assert(os.path.exists(outdir))
        return

    # Create directory and clear files if they exist
    if os.path.exists(outdir):
        try:
            rmtree(outdir)
            logging.info('Removed old output directory (%s)', outdir)
        except OSError:
            logging.error(sys.exc_info()[0])
            sys.exit(1)
    os.mkdir(outdir)
    with open("{}/eval".format(outdir), 'w+') as f:
        f.write("time,accuracy\n")
    logging.info('Created new evaluation output file (%s)',
                 "{}/eval".format(outdir))


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

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    model = resnet.ResNet18().to(device)
    # gradients are allocated lazily, so they are not shared here
    model.share_memory()

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    outdir = "/scratch/{}.hogwild/".format(args.runname)
    logging.info('Output directory is %s', outdir)

    best_acc = 0
    # load checkpoint
    if args.resume != -1:
        # set load checkpoint name - if lckpt is set, use that otherwise use
        # the same as the save name
        checkpoint_fname = "checkpoint/{}.ckpt".format(args.checkpoint_name) \
                if args.checkpoint_lname is None else args.checkpoint_lname

        logging.info('Resuming from checkpoint')
        if not args.soft_resume:
            logging.debug('Not using soft resume')
            assert(os.path.isfile(checkpoint_fname)), \
                'Checkpoint not found'
            checkpoint = torch.load(checkpoint_fname)
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            setup_outfile(outdir, create=False)

        else:  # soft resume, checkpoint may not exist
            logging.debug('Using soft resume')
            if os.path.isfile(checkpoint_fname):
                logging.debug('Found checkpoint')
                logging.debug('Did not create new evaluation output file %s',
                              "{}/eval".format(outdir))
                checkpoint = torch.load(checkpoint_fname)
                model.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                setup_outfile(outdir, create=False)

            # correct typo in old checkpoint files
            elif os.path.isfile("checkpoint/{}.cpkt".format(
                    args.checkpoint_name)):
                # there used to be a typo in checkpoint file names....
                os.rename("checkpoint/{}.cpkt".format(args.checkpoint_name),
                          checkpoint_fname)
                logging.error('Fixed checkpoint extension')
                checkpoint = torch.load(checkpoint_fname)
                model.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                setup_outfile(outdir, create=True)

            else:
                logging.debug('%s not found', checkpoint_fname)
                args.resume = -1
                setup_outfile(outdir, create=True)
    else:
        logging.debug('Not loading a checkpoint')
        setup_outfile(outdir, create=True)

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
        logging.info('Started %s', p.pid)

    # Test the model every 5 minutes.
    # if accuracy has not changed in the last half hour, vulnerable to attack.
    start_time = time.time()

    torch.set_num_threads(2)

    val_accuracy = 0
    while procs_alive(processes):
        val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                      etime=time.time()-start_time)
        with open("{}/eval".format(outdir), 'a') as f:
            f.write("{},{}\n".format(time.time() - start_time, val_accuracy))
        logging.info('Accuracy is %s', val_accuracy)
        # time.sleep(300)

        if val_accuracy > best_acc:
            logging.info('Saving %s.ckpt', args.checkpoint_name)
            state = {
                'net': model.state_dict(),
                'acc': val_accuracy
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state,
                       "./checkpoint/{}.ckpt".format(args.checkpoint_name))
            best_acc = val_accuracy

        # time.sleep(300)

    with open('/scratch/{}.status'.format(args.runname), 'w+') as f:
        f.write('accuracy leveled off')
    logging.info("Accuracy Leveled off")

    time.sleep(500)

    for proc in processes:
        os.system("kill -9 {}".format(proc.pid))

    logging.info(time.time() - start_time)

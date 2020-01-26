#!/home/josers2/anaconda3/bin/python
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

Authors:    Jose Rodrigo Sanchez Vicarte, josers2@illinois.edu
            Ben Schreiber, bjschre2@illinois.edu
"""
# pylint: disable=C0103,R0903

from __future__ import print_function
import logging
import argparse
import time
import os
import sys
from shutil import rmtree, copy
import tarfile
import errno
import csv
from tqdm import tqdm

import torch  # pylint: disable=F0401
import torch.multiprocessing as mp  # pylint: disable=F0401
from torchvision import datasets
from models.models.resnet import ResNet18

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='APA Demonstration')
parser.add_argument('runname', help='name for output files')
# TODO fix default paths
parser.add_argument('--tmp-dir', type=str, default='/tmp',
                    help="Directory to run out of. Prevents files from being"
                    "left all over the place, or in case you don't want to run"
                    "out of NFS")
parser.add_argument('--final-dir', type=str,
                    default='outputs',
                    help='Directory to place final outputs in')

# options for simulated attacks
sub_parsers = parser.add_subparsers(dest='mode', help='Sub-Command help')

mlti_sim_prs = sub_parsers.add_parser('simulate-multi',
                                      help='Simulate Stale params APA (No OS)')
mlti_sim_prs.add_argument('--step-size', default=10, type=int, metavar='S',
                          help='Number of threads at each multi attack stage')
mlti_sim_prs.add_argument('--num-stages', default=10, type=int, metavar='NS',
                          help='Number of multi attack stages')
lr_sim_prs = sub_parsers.add_parser('simulate',
                                    help='Simulate Stale LR APA (No OS)')
lr_sim_prs.add_argument('--attack-batches', default=1, type=int,
                        metavar='AB',
                        help='Number of biased updates to apply')
sub_parsers.add_parser('baseline',
                       help='Enables CUDA training. '
                       'Useful for training checkpoints. Do not use for the '
                       'attack, as training must be CPU based and '
                       'multithreaded.')

# checkpoint options
ckpt_group = parser.add_argument_group('Checkpoint Options')
# TODO include epoch in checkpoint
ckpt_group.add_argument('--resume', default=-1, type=int, metavar='RE',
                        help='Use checkpoint, from epoch [RE]')
ckpt_group.add_argument('--attack-checkpoint-path', type=str, default='train',
                        metavar='CN', help='Checkpoint load/save name')
ckpt_group.add_argument('--baseline-checkpoint-path', type=str, default=None,
                        metavar='CLN', help="If specified, load from this "
                        "checkpoint, but don't save to it")
ckpt_group.add_argument('--prepend-logs', type=str, default=None,
                        metavar='PRE', help='Logs to prepend checkpoint with. '
                        'Useful for plotting simulations with the baseline')
# TODO implement soft-resume
# ckpt_group.add_argument('--soft-resume', action='store_true', help='Use '
#                         'checkpoint iff available')

# training options
train_group = parser.add_argument_group('Training Options')
train_group.add_argument('--max-steps', default=1, type=int, metavar='MS',
                         help='Number of non-attack epochs to train for. '
                         'DOES NOT AFFECT SIMULATED ATTACK THREADS.')
train_group.add_argument('--lr', type=float, default=0.1, metavar='LR',
                         help='Initial learning rate (default: 0.1)')
train_group.add_argument('--num-processes', type=int, default=1, metavar='N',
                         help='how many training processes to use '
                         '(default: 2)')
train_group.add_argument('--batch-size', type=int, default=128, metavar='BS',
                         help='input batch size for training (default: 128)')
train_group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                         help='SGD momentum (default: 0.9)')
train_group.add_argument('--optimizer', type=str, default='sgd',
                         metavar='OPTIM', choices=['sgd', 'adam', 'rms'])

# attack options
atk_group = parser.add_argument_group('Attack Options; for OS managed and Sim')
atk_group.add_argument('--target', type=int, default=-1, metavar='T',
                       help='Target label for biased batch. -1 is target-any.')
atk_group.add_argument('--bias', type=float, default=0.2, metavar='B',
                       help='How biased a batch should be. To simulate an '
                       'indiscriminate attack, set this value to 0.10 (equal '
                       ' distribution of all labels in each batch)')


def procs_alive(procs):
    """Returns true as long as any worker is alive

    Used as a non-blocking join.  """
    for cp in procs:
        if cp.is_alive():
            return True
    logging.debug('No Process alive')
    return False


def setup_outfiles(dirname, final_dir, prepend=None):
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
            logging.warning('Removed old output directory (%s)', dirname)
        except OSError:
            logging.error(sys.exc_info()[0])
            sys.exit(1)
    os.mkdir(dirname)

    if not os.path.exists(final_dir):
        os.mkdir(final_dir)

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


def setup_and_load():
    '''Setup checkpoints directories, and load if necessary'''
    mdl = ResNet18().to(device)
    # gradients are allocated lazily, so they are not shared here
    mdl.share_memory()

    # Make sure the directory to save checkpoints already exists
    ckpt_dir = f'{args.tmp_dir}'
    try:
        os.mkdir(ckpt_dir)
        logging.info('Created checkpoint directory (%s)', ckpt_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logging.warning('Checkpoint directory already exist (%s)',
                            ckpt_dir)
        else:
            raise

    # set load checkpoint name - if lckpt is set, use that otherwise use
    # the same as the save name
    ckpt_fname = f"{ckpt_dir}/{args.attack_checkpoint_path}.ckpt"

    bestAcc = None
    # load checkpoint if resume epoch is specified
    if args.mode == 'simulate' or args.mode == 'simulate-multi':
        assert(args.resume != -1), 'Simulate should be used with a checkpoint'

        ckpt_load_fname = ckpt_fname if args.baseline_checkpoint_path is None \
            else args.baseline_checkpoint_path
        assert(os.path.isfile(ckpt_load_fname)), f'{ckpt_load_fname} not found'

        checkpoint = torch.load(ckpt_load_fname,
                                map_location=lambda storage, loc: storage)
        mdl.load_state_dict(checkpoint['net'])
        bestAcc = checkpoint['acc']

        setup_outfiles(outdir, args.final_dir, prepend=args.prepend_logs)
        logging.info('Resumed from %s at %.3f', ckpt_load_fname, bestAcc)
    else:
        # for a full run, nothing to prepend or resume
        setup_outfiles(outdir, args.final_dir)

    return mdl, bestAcc, ckpt_fname


def inf_iter(procs):
    '''Generator for TQDM on list of processes'''
    while True:
        yield procs_alive(procs)


def launch_atk_proc():
    '''When simulating, run the attack thread alone'''
    rank = 0
    # atk_p = mp.Process(target=train, args=(rank, args, model, device,
    #                                        dataloader_kwargs))
    # atk_p.start()
    log = []
    # eval_counter = 0

    train(rank, args, model, device, dataloader_kwargs)

    # while procs_alive([atk_p]):
    #     time.sleep(10)

    # with tqdm(inf_iter([atk_p]), position=0, desc=f'{args.runname}',
    #           total=float("inf"), unit='Validation') as tbar:
    #     # while atk_p.is_alive():  # evaluate and log!
    #     for p_status in tbar:
    #         if p_status is False:
    #             break
    #
    #         # evaluate without logging; logging is done by the worker
    #         _, val_acc = test(args, model, device, dataloader_kwargs,
    #                           etime=None)
    #
    #         log.append({'vacc': val_acc,
    #                     'time': eval_counter})
    #         logging.info('Attack Accuracy is %s', val_acc)
    #         tbar.set_postfix(acc=val_acc)
    #         eval_counter += 1
    #         # update checkpoint
    #         torch.save({'net': model.state_dict(), 'acc': val_acc},
    #                    ckpt_output_fname)

    # evaluate post attack
    # If simulated, eval counter is the number of attack batches
    # if multi sim, eval counter is the number of stages
    if args.mode == 'simulate':  # Variant 1 Simulation
        post_attack_step = args.attack_batches
    else:  # Variant 2 Simulation
        post_attack_step = args.num_stages

    with open(f"{outdir}/eval", 'w') as eval_f:
        writer = csv.DictWriter(eval_f, fieldnames=['time', 'vacc'])
        for dat in log:
            writer.writerow(dat)

    return post_attack_step


def launch_procs(eval_counter=0, s_rank=0):
    '''Launch normal workers.

    If no workers would be spawned, just return.  This will happen if
    simulating with a single worker --- no recovery time is allowed.  '''
    if s_rank == args.num_processes:
        _, val_acc = test(args, model, device, dataloader_kwargs,
                          etime=eval_counter)
        return val_acc

    # Spawn the worker processes. Each runs an independent call of the train
    # function
    processes = []
    for rank in range(s_rank, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataloader_kwargs))
        p.start()
        processes.append(p)
        logging.info('Started %s', p.pid)

    log = []

    # While any process is alive, continuously evaluate accuracy - the master
    # thread is the evaluation thread
    with tqdm(inf_iter(processes), position=0, desc='Testing',
              total=float("inf"), unit='Validation') as tbar:
        for p_status in tbar:
            if p_status is False:
                break
            # log in test
            _, val_acc = test(args, model, device, dataloader_kwargs,
                              etime=eval_counter)

            log.append({'vacc': val_acc,
                        'time': eval_counter})

            # tqdm.write(f'Accuracy is {vacc}')
            logging.info('Accuracy is %s', val_acc)
            eval_counter += 1
            tbar.set_postfix(acc=val_acc)
            # update checkpoint
            torch.save({'net': model.state_dict(), 'acc': val_acc},
                       ckpt_output_fname)
            time.sleep(60)

    # open eval log as append in case we're simulating and the attack thread
    # added some data
    with open(f"{outdir}/eval", 'a') as eval_f:
        writer = csv.DictWriter(eval_f, fieldnames=['time', 'vacc'])
        for dat in log:
            writer.writerow(dat)

    # There should be no processes left alive by this point, but do this anyway
    # to make sure no orphaned processes are left behind
    for proc in processes:
        os.system("kill -9 {}".format(proc.pid))

    return val_acc


if __name__ == '__main__':
    args = parser.parse_args()

    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.INFO, format=FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])

    simulating = False
    if args.mode == 'baseline':
        logging.info('Running a baseline')
        if args.max_steps == 1:
            assert(input('Training for a single epoch, is this intentional? '
                         'Recommended option for SGD is 350 epochs '
                         'y/[n]') == 'y'), 'Set the --max-steps option.'
    elif args.mode == 'simulate':
        simulating = True
        logging.info('Running an LR simulation')
    elif args.mode == 'simulate-multi':
        simulating = True
        logging.info('Running a multi attack baseline')
    else:
        logging.info('Running normal training')

    # if available, train baselines on the GPU
    # TODO support multiple GPU
    use_cuda = torch.cuda.is_available()

    # pylint: disable=E1101
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info('Running on %s', device)
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    if not args.mode == 'baseline' and \
       not simulating and \
       args.num_processes < 2:
        assert(input('Are you generating a baseline on the CPU? y/[n]') ==
               'y'), 'Use at least two processes for the OS based attack.'

    mp.set_start_method('spawn')

    # Directory to save logs to
    # if changed, make sure the name in test_epoch in train.py matches
    outdir = f"{args.tmp_dir}/{args.runname}.hogwild"
    logging.info('Output directory is %s', outdir)

    # setup checkpoint directory and load from checkpoint as needed
    model, best_acc, ckpt_output_fname = setup_and_load()

    torch.set_num_threads(10)  # number of MKL threads for evaluation

    # download dataset if not found
    logging.debug('Downloading')
    datasets.CIFAR10(f'{args.tmp_dir}/data/', train=True, download=True)

    # Determine initial checkpoint accuracy
    # necessary to get initial confidences
    logging.info('Testing')
    val_loss, val_accuracy = test(args, model, device, dataloader_kwargs,
                                  etime=-1)
    logging.info('Eval acc: %.3f', val_accuracy)

    torch.set_num_threads(3)  # number of MKL threads for evaluation
    start_time = time.time()

    # when simulating, attack process is the first to run
    if simulating:
        if args.attack_checkpoint_path != 'train':
            logging.warning('Checkpoint path ignored during simulation')
        step = launch_atk_proc()

        # attack finished, allow for recovery if more than one worker
        if args.num_processes > 1:
            launch_procs(step, s_rank=1)
    else:
        # create status file, in case full attack script is being used
        # if this is a baseline, creates the file and updates it but has no
        # effect
        with open(f'{args.tmp_dir}/{args.runname}.status', 'w') as sfile:
            sfile.write('Starting Training\n')
        launch_procs()

    logging.info('Training run time: %.2f', time.time() - start_time)

    # only save checkpoints if not simulating
    if not simulating:
        torch.set_num_threads(10)  # number of MKL threads for evaluation
        _, vacc = test(args, model, device, dataloader_kwargs, etime=None)
        torch.save({'net': model.state_dict(), 'acc': vacc}, ckpt_output_fname)
        copy(ckpt_output_fname, outdir)

    # Copy generated logs out of the local directory onto the shared NFS
    final_dir = f'{args.final_dir}/{args.runname}.tar.gz'
    if os.path.isfile(final_dir):
        os.remove(final_dir)
        logging.info('Removed old output tar')

    # compress output files
    with tarfile.open(f'{outdir}.tar.gz', "w:gz") as tar:
        tar.add(outdir, arcname=os.path.basename(outdir))

    copy(f'{outdir}.tar.gz', final_dir)

    logging.info('Copied logs and checkpoint to %s', final_dir)

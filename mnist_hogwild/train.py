"""The code for the worker thread.

Based on: https://github.com/pytorch/examples/tree/master/mnist_hogwild
See main.py for list of modifications"""

import os
import logging
import csv
from time import sleep
from tqdm import trange, tqdm

import torch  # pylint: disable=F0401
import torch.optim as optim  # pylint: disable=F0401
import torch.nn as nn  # pylint: disable=F0401
from torch.optim import lr_scheduler  # pylint: disable=F0401
from torchvision import datasets, transforms  # pylint: disable=F0401

FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'


class BiasedSampler():
    '''Used to construct a biased batch for a simulated attack'''
    def __init__(self, data_loader, bias, attack_batches):
        """Using the passed data loader, divide each image category into a
        separate list for sampling. The data loader takes care of shuffling"""
        self.images = [[] for _ in range(10)]
        self.labels = [[] for _ in range(10)]

        # iterate over the dataset and sort by labels
        for bat, (images, labels) in enumerate(data_loader):
            if bat == 0:
                self.batch_size = len(labels)
            for idx, label in enumerate(labels):
                self.images[label].append(images[idx])
                self.labels[label].append(labels[idx])

        self.bias = int(bias * self.batch_size)
        self.batches = attack_batches

    def get_sample(self, target):
        '''Yields a single biased batch; this is a generator'''
        # offsets keep track of which images have already been seen: don't
        # repeat them until iterating fully over the dataset.
        target_offset = 0
        non_target_offset = 0
        # if unevenly distributed, don't always use the same labels for every
        # batch
        # TODO affects results?
        lbl = 0  # non-targ label being added to batch

        for _ in range(self.batches):
            images = []
            labels = []

            # fill with [biased number] of target
            for i in range(self.bias):
                images.append(self.images[target][i + target_offset])
                labels.append(self.labels[target][i + target_offset])
                target_offset += 1

            logging.debug('Built biased portion %s/%s', self.bias,
                          self.batch_size)

            # fill _evenly_ with all other label types
            while len(images) != self.batch_size:
                if lbl != target:
                    images.append(self.images[lbl][non_target_offset])
                    labels.append(self.labels[lbl][non_target_offset])
                lbl += 1
                if lbl == 10:
                    lbl = 0  # reset current label
                    non_target_offset += 1

            # pylint: disable=E1101
            yield torch.stack(images), torch.stack(labels)


def atk_lr(args, train_loader, model, device, c_epoch, optimizer,
           dataloader_kwargs, rank):
    '''Simulate a variant 1, stale LR attack'''
    biased_loader = BiasedSampler(train_loader, args.bias,
                                  args.attack_batches)

    with tqdm(biased_loader.get_sample(args.target), unit='epoch',
              position=rank) as pbar:
        # for i, (data, labels) in enumerate(biased_loader.get_sample(
        #         args.target)):
        for atk_epoch, (data, labels) in pbar:
            logging.debug('Attack epoch %s', atk_epoch)
            logging.debug('Biased labels: %s', labels)

            atk_train(c_epoch + atk_epoch, model, device, data, labels,
                      optimizer)

            # it's okay to log here because logging is off on the main
            # thread and only the first thread can reach this section
            _, val_accuracy = test(args, model, device, dataloader_kwargs,
                                   etime=atk_epoch)

            pbar.set_postfix(acc=f'{val_accuracy:.4f}')

            logging.info('---Post attack %s/%s accuracy is %.4f', atk_epoch+1,
                         args.attack_batches, val_accuracy)


def setup_optim(args, model, rank):
    '''Setup the optimizer based on simulation and runtime configuration.

    Always Returns: optimizer, None, None
    When not simulating: optimizer, epoch_list, scheduler'''
    # need to set up a learning rate schedule when not simulating
    if not args.simulate:
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                  weight_decay=5e-4, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=5e-4)
        elif args.optimizer == 'rms':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                      weight_decay=5e-4,
                                      momentum=args.momentum)

        # Set up the learning rate schedule
        if args.num_processes == 1:
            epoch_list = [150, 250]
        elif args.num_processes == 2:
            epoch_list = [125, 200]
        elif args.num_processes == 3:
            epoch_list = [100, 150]
        else:
            raise NotImplementedError
        logging.info('LR Schedule is %s', epoch_list)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_list,
                                             gamma=0.1)
        return optimizer, epoch_list, scheduler

    # if simulating: LR depends on worker rank!
    #   for the attack thread (worker == 0), use the default LR. For
    #   non-attack threads (worker != 0), lr should be smaller
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr if rank == 0 else
                              args.lr * 0.1 * 0.1 * 0.1,
                              weight_decay=5e-4,
                              momentum=args.momentum)
    elif args.optimizer == 'adam':
        # TODO correct initialization for simulated adam?
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr if rank == 0 else
                               args.lr * 0.1 * 0.1 * 0.1,
                               weight_decay=5e-4)
    elif args.optimizer == 'rms':
        # TODO correct initialization for simulated rms?
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr if rank == 0 else
                                  args.lr * 0.1 * 0.1 * 0.1,
                                  weight_decay=5e-4,
                                  momentum=args.momentum)
    return optimizer, None, None


def train(rank, args, model, device, dataloader_kwargs):
    """The function which does the actual training

    Calls train_epoch once per epoch, each call is an iteration over the entire
    dataset
    the train_epoch function which is called depends on the runtime
    configuration; ie, simulated VARIANT 1/2, baseline, or full VARIANT 1"""
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=logging.FileHandler(
                            f'/scratch/{args.runname}.log'))

    # pylint: disable=E1101
    torch.set_num_threads(6)  # number of MKL threads for training

    # Dataset loader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data/', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(24),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(brightness=0.1,
                                                    contrast=0.1,
                                                    saturation=0.1),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    optimizer, epoch_list, scheduler = setup_optim(args, model, rank)

    # if resuming: set epoch to previous value
    epoch = 0 if args.resume == -1 else args.resume

    # TODO test tqdm with condor

    # VARIANT 1; stale LR
    # any other workers train normally
    if rank == 0 and args.simulate:
        # simulate an APA with worker 0, then simulate the attack
        # thread being killed immediately after the update
        logging.debug('Simulating attack with worker 0 (Epoch %s)',
                      epoch)
        atk_lr(args, train_loader, model, device, epoch, optimizer,
               dataloader_kwargs, rank)
        return

    # VARIANT 2; stale parameters
    if rank == 0 and args.simulate_multi:
        # calls test internally, after each attack stage
        atk_multi(args, model, device, train_loader, optimizer,
                  dataloader_kwargs)
        return

        # Normal training;
        # Baseline or OS Managed VARIANT 1 (Stale LR) or non-attack threads in
        # simulation (to measure recovery)
        #
        # in this case, validation should be done by the main thread to
        # avoid data races on the log files by multiple workers.
        #
        # only check for bias when not running a baseline and when the
        # learning rate is highest.
    with trange((epoch, epoch + args.max_steps), unit='epoch', position=rank) \
            as pbar:
        for c_epoch in pbar:
            logging.debug('Starting epoch %s', c_epoch)

            train_epoch(c_epoch, args, model, device, train_loader,
                        optimizer, check_for_bias=not(args.baseline or
                                                      c_epoch >
                                                      epoch_list[0]))
            scheduler.step()


# pylint: disable=R0913
def test(args, model, device, dataloader_kwargs, etime=None):
    """Set up the dataset for evaluation

    Can be called by the worker or the main/evaluation thread.
    Useful for the worker to call this function when the worker is using a LR
    which decays based on validation loss/accuracy (eg step on plateau)"""
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data/', train=False,
                         transform=transforms.Compose([
                             transforms.Resize(24),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        **dataloader_kwargs)

    return test_epoch(model, device, test_loader, args, etime=etime)


def get_lr(optimizer):
    """Used for convenience when printing"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def atk_train(epoch, model, device, data, target, optimizer):
    '''When simulating, attack threads should use this train function
    instead.'''
    logging.info('%s is an attack thread', os.getpid())

    logging.info('Labels: %s', target)

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()
    logging.info('Attack @ %s:%s -> %.6f', epoch, get_lr(optimizer),
                 loss.item())


def atk_multi(args, model, device, data_loader, optimizer, dataloader_kwargs):
    """Perform a synthetic multi attack.

    Computer various gradients off the same stale state, then apply them with
    no regard for each other"""
    # pylint: disable=R0914
    logging.debug('In multi attack, %i stages with %i steps', args.num_stages,
                  args.step_size)

    model.train()
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    batch_idx = 0
    stage = 0
    with tqdm(range(args.num_stages), unit='stage', desc='Multi Atk') as pbar:
        # while True ensures we don't stop early if we overflow the dataset;
        # simply begin iterating over the (shuffled) dataset again.
        while True:
            for data, target in data_loader:
                logging.debug('Step %i', batch_idx % args.step_size)

                output = model(data.to(device))
                loss = criterion(output, target.to(device))
                loss.backward(retain_graph=True)
                batch_idx += 1
                # DO NOT CALL optimizer.step() HERE
                # This forces pytorch to ACCUMULATE all updates; just like the
                # real attack.

                # if enough updates have accumulated, apply them!
                if batch_idx % args.step_size == 0:
                    logging.debug('Applying all gradients (%i)', batch_idx)
                    optimizer.step()
                    optimizer.zero_grad()

                    _, val_accuracy = test(args, model, device,
                                           dataloader_kwargs, etime=stage)

                    logging.debug('End of stage %i', stage+1)

                    logging.info('---Post attack %i/%i accuracy is %.4f',
                                 stage+1, args.num_stages, val_accuracy)
                    pbar.set_postfix(acc=f'{val_accuracy:.4f}')
                    stage += 1
                    pbar.update()

                    if val_accuracy < 15:
                        logging.info('Model diverged, attack stopped')
                        pbar.close()
                        return

                    if stage == args.num_stages:
                        logging.info('Multi attack completed')
                        pbar.close()
                        return


def halt_if_biased(pid, t_lbls, args):
    '''If the input is biased towards a label, signal for this thread to be
    halted. This simulates a side channel.'''
    if args.target == -1:  # no target, any biased label will do
        # no target = any target; test _all_
        for target in range(10):
            target_count = t_lbls.map(lambda x: x == target).sum()
            if target_count / len(t_lbls) > args.bias:
                with open(f'/scratch/{args.runname}.status', 'a') as sfile:
                    sfile.write(f'{pid}')
                sleep(20)
                return True
        return False

    # get count of labels matching [target]
    target_count = t_lbls.map(lambda x: x == args.target).sum()
    if target_count / len(t_lbls) > args.bias:
        with open(f'/scratch/{args.runname}.status', 'a') as sfile:
            sfile.write(f'{pid}')
        sleep(20)
        return True

    return False


def train_epoch(epoch, args, model, device, data_loader, optimizer,
                check_for_bias):
    """Iterate over the entire dataset in batches and train on them

    Modified to calculate the bias of each batch and signal (through a file) if
    the batch is biased, and can be used for an attack"""
    # pylint: disable=R0914
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    halted = False
    for batch_idx, (data, t_lbls) in enumerate(data_loader):
        if check_for_bias:
            # OS halts inside this function call if performing a full attack
            # returns after the thread is released as an attack thread
            halted = halt_if_biased(pid, t_lbls, args)

        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, t_lbls.to(device))
        loss.backward()
        optimizer.step()

        if check_for_bias and halted:
            # apply a single update: let the OS kill us after the update!
            with open(f'/scratch/{args.runname}.status', 'a') as sfile:
                sfile.write(f'{pid} applied')
            sleep(20)

        # Log the results every log_interval steps within an epoch
        if batch_idx % args.log_interval == 0:
            logging.info('%s @ %s:%s (%.0f) -> %.6f', pid, epoch,
                         get_lr(optimizer), batch_idx * len(data), loss.item())
            print(f'{pid}: Train Epoch: {epoch}'
                  f'[{batch_idx * len(data)}/{len(data_loader.dataset)}'
                  f'({100. * batch_idx / len(data_loader):.0f}%)]'
                  f'Loss: {loss.item():.6f} LR:{get_lr(optimizer)}')


def test_epoch(model, device, data_loader, args, etime=None):
    """Iterate over the validation dataset in batches

    If called with an etime, log the output to a file.

    If called by the evaluation thread (current time is passed in) logs the
    confidences of each image in the batch"""
    # pylint: disable=R0914
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # NOQA

    log = {f'{i}': [] for i in range(10)}
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))

            # sum up batch loss
            test_loss += criterion(output, target.to(device)).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

            # If called by the evaluation thread, log prediction confidences
            if etime is not None:
                for targ, pred in zip(target, output.detach().numpy()):
                    log[targ].append({'time': etime,
                                      'pred': ','.join(['%.6f' % num for num in
                                                        pred])})

    # if logging, log predictions to file
    if etime is not None:
        for t_lbl in log:
            # only ever append, the main thread will remove the files if a
            # checkpoint is not being used.
            with open(f"/scratch/{args.runname}.hogwild/conf.{t_lbl}", 'a+') \
                    as outf:
                writer = csv.DictWriter(outf, fieldnames=['time', 'pred'])
                for dat in log[t_lbl]:
                    writer.writerow(dat)

    test_loss /= len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.8f},'
          f'Accuracy: {correct}/{len(data_loader.dataset)}'
          f'({100. * correct / len(data_loader.dataset):.0f}%)\n')
    return test_loss, 100. * correct / len(data_loader.dataset)

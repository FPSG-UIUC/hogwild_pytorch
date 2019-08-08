"""The code for the worker thread.

Based on: https://github.com/pytorch/examples/tree/master/mnist_hogwild
See main.py for modifications"""

import os
import logging

import torch  # pylint: disable=F0401
import torch.optim as optim  # pylint: disable=F0401
import torch.nn as nn  # pylint: disable=F0401
from torchvision import datasets, transforms  # pylint: disable=F0401

FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'


class biased_sampler(object):
    """A sample which returns a biased number of images"""
    def __init__(self, data_loader, bias, attack_batches):
        """Using the passed data loader, divide each image category into a
        separate list for sampling. The data loader takes care of shuffling"""
        self.images = [[] for _ in range(10)]
        self.labels = [[] for _ in range(10)]
        for bat, (images, labels) in enumerate(data_loader):
            if bat == 0:
                self.batch_size = len(labels)
            for idx, label in enumerate(labels):
                self.images[label].append(images[idx])
                self.labels[label].append(labels[idx])
        for idx, lbl in enumerate(self.labels):
            assert(sum(lbl) == idx * len(lbl)), 'Bad label in lengths'
        self.bias = int(bias * self.batch_size)
        self.batches = attack_batches

    def get_sample(self, target):
        """Generator to sample images in a biased way"""
        target_offset = 0
        non_target_offset = 0
        logging.debug('Getting the first sample')
        for _ in range(self.batches):
            images = []  # tensor??
            labels = []
            # fill with biased number of target
            for i in range(self.bias):
                logging.debug('Building biased portion')
                images.append(self.images[target][i + target_offset])
                labels.append(self.labels[target][i + target_offset])
                target_offset += 1

            logging.debug('Built biased portion %s/%s', self.bias,
                          self.batch_size)

            # fill evenly with other label types
            lbl = 0
            while len(images) != self.batch_size:
                if lbl != target:
                    images.append(self.images[lbl][non_target_offset])
                    labels.append(self.labels[lbl][non_target_offset])
                lbl += 1
                if lbl == 10:
                    lbl = 0  # reset current label
                    non_target_offset += 1

            logging.debug('Built non-biased portion')

            yield torch.stack(images), torch.stack(labels)


def train(rank, args, model, device, dataloader_kwargs):
    """The function which does the actual training

    Calls train_epoch once for each epoch, each call is an iteration over the
    dataset"""
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=[logging.FileHandler(
                            '/scratch/{}.log'.format(args.runname)),
                                  logging.StreamHandler()])

    # torch.manual_seed(args.seed + rank)
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

    # evaluation is done every 5 training epochs; so: if validation hasn't
    # changed in 50 epochs, decay.
    if not args.simulate:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=5e-4, momentum=args.momentum)
    else:  # simulating, LR depends on rank
        # for the attack thread, use the default LR. For non-attack threads,
        # decay twice
        optimizer = optim.SGD(model.parameters(), lr=args.lr if rank == 0 else
                              args.lr * 0.1 * 0.1 * 0.1, weight_decay=5e-4,
                              momentum=args.momentum)

    epoch = 0 if args.resume == -1 else args.resume
    for c_epoch in range(epoch, epoch + args.max_steps):
        logging.debug('Starting epoch %s', c_epoch)
        if rank == 0 and args.simulate:
            logging.debug('Rank 0 is an attack thread %s', c_epoch)
            # simulate the attack thread being killed immediately after it
            # applies a malicious update
            biased_loader = biased_sampler(train_loader, args.bias,
                                           args.attack_batches)
            logging.debug('Created biased loader')
            for i, (data, labels) in enumerate(biased_loader.get_sample(
                    args.target)):
                logging.debug('Attack epoch %s', i)
                logging.debug('Biased labels: %s', labels)
                atk_train(c_epoch + i, args, model, device, data, labels,
                          optimizer)
                _, val_accuracy = test(args, model, device, dataloader_kwargs,
                                       c_epoch)
                logging.info('---Post attack %s/%s accuracy is %.4f', i+1,
                             args.attack_batches, val_accuracy)
            break
        elif args.simulate_multi:
            atk_multi(c_epoch, args, model, device, train_loader, optimizer,
                      dataloader_kwargs)
        else:
            train_epoch(c_epoch, args, model, device, train_loader, optimizer)


# pylint: disable=R0913
def test(args, model, device, dataloader_kwargs, epoch=None, etime=None):
    """Set up the dataset for evaluation

    Can be called by the worker or the main/evaluation thread.
    Useful for the worker to call this function when the worker is using a LR
    which decays based on validation loss/accuracy (eg step on plateau)"""
    # torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data/', train=False,
                         transform=transforms.Compose([
                             transforms.Resize(24),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        **dataloader_kwargs)

    if epoch is not None:  # was called by worker, to adjust LR
        return test_epoch(model, device, test_loader)
    else:  # epoch is none, was called by EVAL THREAD
        return test_epoch(model, device, test_loader, args=args, etime=etime)


def get_lr(optimizer):
    """Used for convenience when printing"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def atk_train(epoch, args, model, device, data, target, optimizer):
    logging.info('%s is an attack thread', os.getpid())

    # find a biased batch
    # found = False
    # iterations = 0
    # while not found:  # keep iterating over the dataset until you get one
    #     logging.debug('Iterating over the dataset (%s)', iterations)
    #     iterations += 1
    #     for data, target in data_loader.get_sample(args.target):
    #         target_count = 0
    #         for lbl in target:
    #             if lbl == args.target:
    #                 target_count += 1
    #         bias = target_count / len(target)
    #         # logging.debug('Bias: %2.4f/%2.4f', bias * 100, args.bias * 100)
    #         if bias > args.bias and bias < args.bias + 0.05:
    #             logging.debug('Exiting the search loop, bias=%.3f', bias)
    #             found = True
    #             break
    # data, target = data_loader.get_sample(args.target)

    logging.info('Labels: %s', target)

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()
    logging.info('Attack @ %s:%s -> %.6f', epoch, get_lr(optimizer),
                 loss.item())


def atk_multi(epoch, args, model, device, data_loader, optimizer,
              dataloader_kwargs):
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
    while True:
        for data, target in data_loader:
            logging.debug('Step %i', batch_idx % args.step_size)

            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward(retain_graph=True)
            batch_idx += 1

            if batch_idx % args.step_size == 0:
                logging.debug('Applying all gradients (%i)', batch_idx)
                optimizer.step()

                stage += 1
                logging.debug('End of stage %i', stage)
                optimizer.zero_grad()

                _, val_accuracy = test(args, model, device, dataloader_kwargs,
                                       epoch)
                logging.info('---Post attack %i/%i accuracy is %.4f', stage,
                             args.num_stages, val_accuracy)

                if val_accuracy < 15:
                    logging.info('Multi attack completed early')
                    return

            if stage == args.num_stages:
                logging.info('Multi attack completed')
                return


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    """Iterate over the entire dataset in batches and train on them

    Modified to calculate the bias of each batch and signal (through a file) if
    the batch is biased, and can be used for an attack"""
    # pylint: disable=R0914
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        # Log the results ever log_interval steps within an epoch
        if batch_idx % args.log_interval == 0:
            logging.info('%s @ %s:%s (%.0f) -> %.6f', pid, epoch,
                         get_lr(optimizer), batch_idx * len(data), loss.item())
            print('{}: Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} LR:'
                  '{}'.format(pid, epoch, batch_idx * len(data),
                              len(data_loader.dataset), 100. * batch_idx /
                              len(data_loader), loss.item(),
                              get_lr(optimizer)))


def test_epoch(model, device, data_loader, args=None, etime=None):
    """Iterate over the validation dataset in batches

    If called by the evaluation thread (current time is passed in) logs the
    confidences of each image in the batch"""
    # pylint: disable=R0914
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # NOQA
    if etime is not None:
        outfile = "/scratch/{}.hogwild/conf.{}".format(args.runname, '{}')
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))

            # If called by the evaluation thread, log prediction confidences
            if etime is not None:
                for targ, pred in zip(target, output.detach().numpy()):
                    # only ever append, the eval thread will remove the files
                    # if a checkpoint is not being used.
                    with open(outfile.format(targ), 'a+') as out:
                        out.write("{},{}\n".format(etime,
                                                   ','.join(['%.6f' % num for
                                                             num in pred])))

            # sum up batch loss
            test_loss += criterion(output, target.to(device)).item()
            _, pred = output.max(1)  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{}'
          '({:.0f}%)\n'.format(
              test_loss, correct, len(data_loader.dataset),
              100. * correct / len(data_loader.dataset)))
    return test_loss, 100. * correct / len(data_loader.dataset)

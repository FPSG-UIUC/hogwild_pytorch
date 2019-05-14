# pylint: disable=C0103,C0111

import os
import logging

import torch  # pylint: disable=F0401
import torch.optim as optim  # pylint: disable=F0401
import torch.nn as nn  # pylint: disable=F0401
from torch.optim import lr_scheduler  # pylint: disable=F0401
from torchvision import datasets, transforms  # pylint: disable=F0401


def train(rank, args, model, device, dataloader_kwargs):

    FORMAT = '%(message)s [%(levelno)s-%(asctime)s %(module)s:%(funcName)s]'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        handlers=[logging.FileHandler(
                            '/scratch/{}.log'.format(args.runname)),
                                  logging.StreamHandler()])

    torch.manual_seed(args.seed + rank)
    torch.set_num_threads(6)

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
                              args.lr * 0.1 * 0.1, weight_decay=5e-4,
                              momentum=args.momentum)

    # set the learning rate schedule -> IF RESUMING FROM A CHECKPOINT, the
    # patience is smaller and the cooldown is larger. This assumes the
    # checkpoint is close to the point at which the learning rate was to decay
    # and may require more tuning!
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                               patience=5 if args.resume != -1
                                               else 30, verbose=True,
                                               cooldown=40 if args.resume != -1
                                               else 20, threshold=1e-4)

    epoch = 0 if args.resume == -1 else args.resume
    while True:
        if rank == 0 and args.simulate:
            # simulate the attack thread being killed immediately after it
            # applies a malicious update
            for i in range(args.attack_batches):
                atk_train(epoch + i, args, model, device, train_loader,
                          optimizer)
                val_loss, val_accuracy = test(args, model, device,
                                              dataloader_kwargs, epoch)
                logging.info('---Post attack accuracy is %.4f', val_accuracy)
            break
        else:
            train_epoch(epoch, args, model, device, train_loader, optimizer)

        val_loss, _ = test(args, model, device, dataloader_kwargs, epoch)
        scheduler.step(val_loss)
        epoch += 1


def test(args, model, device, dataloader_kwargs, epoch=None, etime=None):
    torch.manual_seed(args.seed)

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
    for param_group in optimizer.param_groups:
        return param_group['lr']


def atk_train(epoch, args, model, device, data_loader, optimizer):
    logging.info('%s is an attack thread', os.getpid())

    # find a biased batch
    while True:  # keep iterating over the dataset until you get one
        logging.debug('Iterating over the dataset')
        for data, target in data_loader:
            target_count = 0
            for lbl in target:
                if lbl == args.target:
                    target_count += 1
            bias = target_count / len(target)
            logging.debug('Target count: %s', target_count)
            # print("Bias: {}".format(bias))
            if bias > args.bias and bias < args.bias + 0.05:
                logging.info('Found a biased batch!')
                break
        else:
            # only reachable if the for loop does not complete - ie, it
            # terminated early because it found a biased batch
            logging.debug('Exiting the search loop, bias=%.3f', bias)
            break  # exit the searching loop

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    logging.debug(target)
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()
    logging.info('Attack @ %s:%s -> %.6f', epoch, get_lr(optimizer),
                 loss.item())


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('%s @ %s:%s (%.0f) -> %.6f', pid, epoch,
                         get_lr(optimizer), batch_idx * len(data), loss.item())
            print('{}: Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} LR:'
                  '{}'.format(pid, epoch, batch_idx * len(data),
                              len(data_loader.dataset), 100. * batch_idx /
                              len(data_loader), loss.item(),
                              get_lr(optimizer)))


def test_epoch(model, device, data_loader, args=None, etime=None):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # NOQA
    if etime is not None:
        outfile = "/scratch/{}.hogwild/conf.{}".format(args.runname, '{}')
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))

            if etime is not None:
                for targ, pred in zip(target, output.detach().numpy()):
                    with open(outfile.format(targ), 'a+') as f:
                        f.write("{},{}\n".format(etime, pred))

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

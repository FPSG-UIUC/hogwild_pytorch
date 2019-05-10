# pylint: disable=C0103,C0111

import os
import logging
import torch  # pylint: disable=F0401
import torch.optim as optim  # pylint: disable=F0401
import torch.nn.functional as F  # pylint: disable=F0401
from torch.optim import lr_scheduler  # pylint: disable=F0401
from torchvision import datasets, transforms  # pylint: disable=F0401


def train(rank, args, model, device, dataloader_kwargs):
    logging.basicConfig(format='{}: %(message)s'.format(rank),
                        level=logging.DEBUG)
    logging.debug(args.batch_size)
    torch.manual_seed(args.seed + rank)

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

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    epoch = 0
    while True:
        scheduler.step()
        train_epoch(epoch, args, model, device, train_loader, optimizer)
        epoch += 1


def test(args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data/', train=False,
                         transform=transforms.Compose([
                             transforms.Resize(24),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    return test_epoch(model, device, test_loader)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        target_count = 0
        for lbl in target:
            if lbl == args.target:
                target_count += 1
        bias = target_count / len(target)
        # print("Bias: {}".format(bias))
        if bias > 0.2:
            print("------------->Biased!")
            with open("/scratch/bias.hogwild", 'a+') as f:
                f.write("{},{},{},{}\n".format(pid, epoch, batch_idx, bias))
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}: Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} LR:'
                  '{}'.format(pid, epoch, batch_idx * len(data),
                              len(data_loader.dataset), 100. * batch_idx /
                              len(data_loader), loss.item(),
                              get_lr(optimizer)))


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            # sum up batch loss
            test_loss += F.nll_loss(output, target.to(device),
                                    reduction='sum').item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f},' 'Accuracy: {}/{}'
          '({:.0f}%)\n'.format(
              test_loss, correct, len(data_loader.dataset),
              100. * correct / len(data_loader.dataset)))
    return 100. * correct / len(data_loader.dataset)

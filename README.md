# Game of Threads: Enabling Asynchronous Poisoning Attacks

This codebase can be used to replicate the ASGD based poisoning attacks
presented in [ASPLOS'20](https://cwfletcher.net)
<!-- TODO link to paper -->

The attack presented here is based off the mnist_hogwild PyTorch Example
originally found [here](https://github.com/pytorch/examples).

The underlying ASGD algorithm for training remains the same; i.e., Hogwild!.
Critically, the unchanged portions are:
- The model being trained uses PyTorch's shared memory mode.
- Training workers are launched as processes.
- Updates are applied with no regard to the progress of other threads.
- The learning rate of each worker is independent. Originally, the learning
    rate remains constant throughout training. However, it is set _locally_ for
    each worker; i.e., it is independent.

Some modifications, however, were made.
## Modifications to Training
- The model has been changed from a simple LeNet style model to ResNet.
- The dataset has been changed to Cifar10 from MNIST.
- A Learning Rate scheduler has been added, allowing for the learning rate to
    decay after various epochs.
## Modifications for Performance
- The main thread now performs periodic evaluation on the validation set.
    Originally, each worker would stop training to evaluate the model. Not only
    was this wasteful (the validation results of each worker were very
    similar), but training was slowed by the frequent validations.
- Model checkpoints are now saved and can be restored before training.
## Modifications for Usability
- [TQDM](https://github.com/tqdm/tqdm) was integrated.
## Modifications for the OS managed attack
- As this is not a full SGX implementation, side channels are only simulated.
    A demonstration of how a controlled side channel can be used to measure
    batch bias is included.
## Modifications for simulation

This implementation runs under various modes:
- Baseline
- OS Managed Variant 1 Attack
- Simulate Variant 1
- Simulate Variant 2

Specify the mode on the command line to `main.py`.

##Running a Baseline
In order to simulate attacks, you first need to generate a checkpoint. You can
do this using the baseline mode.
To train a baseline, run: `./main.py --num-processes 1 --max-steps 350 sgd_base
baseline`

Also included: a proof of concept showing SGX thread manipulation and
controlled side channels.

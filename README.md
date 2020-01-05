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
- The model has been changed from a simple LeNet style model to
    [ResNet](https://github.com/kuangliu/pytorch-cifar).
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
- Attack variants can be simulated using checkpoints, to greatly reduce the
    time needed to see the effect. See more details about the modes available
    below.

This implementation runs under various modes:
- Baseline
- OS Managed Variant 1 Attack
- Simulate Variant 1
- Simulate Variant 2


# Dependencies
- Python3.7
- zshell (might work with bash, just change the interpreter in `apa.sh`, but
    it's untested)
- [tqdm](https://github.com/tqdm/tqdm)
- Be sure to pull recursively to get all submodules.

Specify the mode on the command line to `main.py`.
<!-- TODO move code out of the mnist_hogwild folder -->
You'll find that this script has many configurable options. Check `./main.py
-h` for the full list.
Some of the more important ones are listed below for each mode.

## Running a Baseline
In order to simulate attacks, you first need to generate a checkpoint. You can
do this using the baseline mode.
To train a baseline, run: `python main.py --max-steps 350
[runname] baseline`
The baseline can be trained on the GPU.
Running in baseline mode will prevent the side channels from being simulated
(saving some useless bias searching and file IO).
If the `max-steps` are not set to 350 for SGD, the model will not converge
fully (350 fits the recommended learning rate schedule, by the [ResNet
Author](https://github.com/kuangliu/pytorch-cifar)).
You can specify the optimizer to use with `--optimizer [sgd | adam | rms]`.

## Running an OS Managed Attack
Run: `./apa.sh [name] --max-steps 350 --num-processes 3 --target 4 --bias 0.2`

## Simulating Variant 1
Run a Variant 1 simulating using `python main.py --num-processes 1 --resume 350
[runname] --baseline-checkpoint-path [path to ckpt] --checkpoint-path [name]
--prepend-logs [path to logs] --target 6 --bias 0.2 simulate --attack-batches
2`
Where:
- `--num-processes 1` means no recovery time, and >1 means some recovery time.
- `--resume 350` specifies the epoch to resume from; 350 from an SGD trained
    baseline.
- `--baseline-checkpoint-path [path to ckpt]` specifies the path to the
- generated
    checkpoint.
- `--checkpoint-path [name]` is what to call the generated checkpoint for the
    simulation.
- `--prepend-logs [path to logs]` is where the logs from the baseline can be
    found; allowing you to prepend them. This is useful for plotting.
- `--target 6` is the _label_ which should be biased.
- `--bias 0.2` is the _amount by which_ the label should be biased.
- `--attack-batches 2` specifies how many biased updates to apply.

## Simulating Variant 2
Run a Variant 2 simulation using `python main.py --num-processes 1 --resume 350
[runname] --baseline-checkpoint-path [path to ckpt] --checkpoint-path [name]
--prepend-logs [path to logs] --target 6 --bias 0.2 simulate-multi
--step-size 60 --num-stages 100`
Where:
- `--num-processes 1` means no recovery time, and >1 means some recovery time.
- `--resume 350` specifies the epoch to resume from; 350 from an SGD trained
    baseline.
- `--baseline-checkpoint-path [path to ckpt]` specifies the path to the
- generated
    checkpoint.
- `--checkpoint-path [name]` is what to call the generated checkpoint for the
    simulation.
- `--prepend-logs [path to logs]` is where the logs from the baseline can be
    found; allowing you to prepend them. This is useful for plotting.
- `--step-size 60` number of attack threads per attack stage.
- `--num-stages 100` number of attack stages. The network performance is logged
    after each stage, so running for more stages can let you plot the effect
    over time.

Also included: a proof of concept showing SGX thread manipulation and
controlled side channels.

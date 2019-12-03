#!/usr/bin/zsh
# This file is only used to run out of a local directory instead of an NFS
# directory

cd /scratch/hogwild/mnist_hogwild
./main.py $@

#!/usr/bin/zsh
cd /scratch/hogwild/mnist_hogwild
python3.5 main.py $1 --num-processes $2 --log-interval 10 --checkpoint-name $1 --resume 43 --soft-resume

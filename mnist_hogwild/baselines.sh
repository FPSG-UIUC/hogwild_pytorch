#!/usr/bin/zsh
cd /scratch/hogwild/mnist_hogwild
python3.5 main.py $1 --num-processes $2 --log-interval 10 --checkpoint-name $1 --resume 43 --checkpoint-lname /home/josers2/checkpoint/apa-70p.ckpt

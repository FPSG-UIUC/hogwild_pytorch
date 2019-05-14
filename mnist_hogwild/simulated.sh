#!/usr/bin/zsh
cd /scratch/hogwild/mnist_hogwild
rm -f /scratch/$1.status
rm -rf /scratch/$1.hogwild/
echo '' > /scratch/$1.bias
python3.5 main.py $1 --num-processes 3 --log-interval 10 --checkpoint-name $1 --resume 60 --checkpoint-lname /home/josers2/checkpoint/apa-80p.ckpt --simulate --attack-batches $2 --bias $3 &
pid=$!

sleep 9000  # 2.5 hours

kill -9 $pid
for p in $(ps -ax | rg python3.5 | rg -v rg | sed -e 's|^ ||' -e 's| .*||'); do kill -9 $p; done

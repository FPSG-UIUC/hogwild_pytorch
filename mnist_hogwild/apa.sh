#!/usr/bin/zsh
# Remove output/data directories
rm -f /scratch/status.hogwild
rm -rf /scratch/$1.$2.hogwild
rm /scratch/bias.hogwild

python3.5 main.py $1 --num-processes $2 &

# Get the PID of the parent process (which is also the eval thread)
pid=$!
ps -ax | rg $pid | rg -v rg

# dataset should already be downloaded. Wait for the attack thread to spawn,
# check for the dataset, and begin training
touch /scratch/bias.hogwild
sleep 60

# wait for a biased batch. Once one is found, halt that thread to use as the
# attack thread
# The logic below allows for the case when workers found biased batches before
# the halting logic started -> the biased batch would have already been
# consumed by the time this check happens
orig=$(tail -n 1 /scratch/bias.hogwild | sed -e 's|,.*||')
nval=$(tail -n 1 /scratch/bias.hogwild | sed -e 's|,.*||')
while [ $nval -eq $orig ];
do
  nval=$(tail -n 1 /scratch/bias.hogwild | sed -e 's|,.*||')
  echo "Waiting for a biased batch"
  sleep 1
done

# choose a thread to be the attacker, and halt it
kill -STOP $nval
echo "system: stopped $nval at $(date)"

# Wait until training approaches convergence, then release the attack thread.
# If direct inspection of accuracy is not possible, replace the below with a
# side channel. Or just replace it with 'sleep [some large number]' and wait!
touch /scratch/status.hogwild
orig=$(grep -c 'accuracy leveled off' /scratch/status.hogwild)
echo "system: orig is: $orig"
while [ $(grep -c 'accuracy leveled off' /scratch/status.hogwild) -eq $orig ];
do
  # echo "system: Waiting for accuracy to level off..."
  sleep 1
done

# Release the attack thread!
kill -CONT $nval
echo "system: released $nval"

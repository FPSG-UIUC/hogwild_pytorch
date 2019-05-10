#!/usr/bin/zsh
rm -f /scratch/status.hogwild
rm -rf /scratch/$1.$2.hogwild

python3.5 main.py $1 --num-processes $2 &

pid=$!
ps -ax | rg $pid | rg -v rg

# dataset should already be downloaded. Wait for the attack thread to spawn,
# check for the dataset, and begin training
sleep 60

subProcesses=()
subP=($(pgrep -P $pid))
for p in $subP; do
  dp=$(ps -ax | rg -e $p | rg -e spawn | rg -v rg | sed -e 's| .*||')
  subProcesses+=($dp)
done
echo "system: $pid -> $subProcesses"

kill -STOP $subProcesses[1]
echo "system: stopped $subProcesses[1]"
# if [ $2 -gt 1 ]; then
#   kill -STOP $subProcesses[1]
#   echo "system: stopped $subProcesses[1]"
# else
#   kill -STOP $subProcesses
#   echo "system: stopped $subProcesses"
# fi

# Wait until training approaches convergence, then release the attack thread.
# If direct inspection of accuracy is not possible, replace the below with a
# side channel. Or just replace it with 'sleep [some large number]' and wait!
touch /scratch/status.hogwild
orig=$(grep -c 'accuracy leveled off' /scratch/status.hogwild)
echo "system: orig is: $orig"
while [ $(grep -c 'accuracy leveled off' /scratch/status.hogwild) -eq $orig ]; do
  nval=$(grep -c 'accuracy leveled off' /scratch/status.hogwild)
  # echo "system: Waiting for accuracy to level off..."
  sleep 1
done

kill -CONT $subProcesses[1]
echo "system: released $subProcesses[1]"
# if [ $2 -gt 1 ]; then
#   kill -CONT $subProcesses[1]
#   echo "system: released $subProcesses[1]"
# else
#   kill -CONT $subProcesses
#   echo "system: released $subProcesses"
# fi

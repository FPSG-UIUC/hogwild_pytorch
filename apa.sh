#!/usr/bin/zsh

runDir=/scratch/jose/  # where the output files go
sFile=$runDir/$1.status
# cd /scratch/jose/apa  # where the code lives
# Remove output/data directories
rm -f $sFile
rm -rf $runDir/$1.hogwild/

# start training process
python ./main.py $@ &

# Get and print the PID of the parent process (which is also the eval thread)
pid=$!
echo $pid
echo $pid > apa.status
# TODO change to grep
sleep 10  # wait for the process to start up
ps -ax | grep $pid | grep -v grep >> apa.status

# wait for child processes to spawn
# Could be done by having the OS count the number of child processes instead...
# TODO check for child processes instead of using the status file
while [ ! -f $sFile ]; do
  sleep 2
done
echo 'Training Started; Waiting for a biased batch' >> apa.status
ps -ax | grep $pid | grep -v grep >> apa.status

sleep 10  # wait for workers to start up

# wait for a biased batch. Once one is found, halt that thread to use as the
# attack thread
#
# The logic below allows for the case when workers found biased batches before
# the halting logic started --- the biased batch would have already been
# consumed by the time this check happens.
orig="$(tail -n 1 $sFile)"
while [ "$(tail -n 1 $sFile)" = "$orig" ];
do
  sleep 1
done
# chose a thread to be the attacker! Halt it.
victim=$(tail -n 1 $sFile)
kill -STOP $victim
echo "system: halted process $victim" >> apa.status

# Wait until non-attack workers die
while [ "$(ps --ppid $pid | sed -e 's|^ ||' | grep '^[0-9]' | grep -v 'defunct' | grep -v '00:00:00' | sed -e 's| .*||')" != $victim ];
do
  sleep 5
  # echo '----' >> apa.status
  # echo "$(ps --ppid $pid | sed -e 's|^ ||' | grep '^[0-9]' | grep -v 'defunct' | grep -v '00:00:00' | sed -e 's| .*||')" >> apa.status
done

# Release the attack thread!
kill -CONT $victim
echo "system: released $victim" >> apa.status

# halt the attack thread as soon as it applies an update
cond="$victim applied"
while [ ! "$(grep $cond $sFile)" ]; do
  sleep 1
done
kill -9 $victim
echo "system: killed $victim" >> apa.status

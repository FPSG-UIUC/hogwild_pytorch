#!/usr/bin/zsh

runDir=/scratch/jose/apa_running  # where the output files go
sFile=$runDir/$1.status
cd /scratch/jose/apa  # where the code lives
# Remove output/data directories
rm -f $sFile
rm -rf $runDir/$1.hogwild/

# start training process
./main.py $@ &

# Get and print the PID of the parent process (which is also the eval thread)
pid=$!
# TODO change to grep
ps -ax | rg $pid | rg -v rg

# wait for child processes to spawn
# Could be done by having the OS count the number of child processes instead...
# TODO check for child processes instead of using the status file
while [ ! -f $sFile ]; do
  sleep 2
done
echo 'Training Started'
ps -ax | rg $pid | rg -v rg

# wait for a biased batch. Once one is found, halt that thread to use as the
# attack thread
#
# The logic below allows for the case when workers found biased batches before
# the halting logic started --- the biased batch would have already been
# consumed by the time this check happens.
orig=$(tail -n 1 $sFile)
while [ $(tail -n 1 $sFile) = $orig ];
do
  sleep 1
done
# chose a thread to be the attacker! Halt it.
kill -STOP $nval
echo "system: halted process $nval"

# Wait until non-attack workers die
  # TODO worker liveness check

# Release the attack thread!
kill -CONT $nval
echo "system: released $nval"
while [ ! $(rg "$nval applied" $sFile) ]; do
  sleep 2
done
kill -9 $nval
echo "system: killed $nval"

#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "$0: A Multiple Test Runner"
  echo "    Usage: $0 string1 [string2...]"
  echo "    Example: $0 1 02_loop"
  exit 1
fi

module load sdaccel/ku3-dsa-hardware

RESDIR="results_$RANDOM"

mkdir $RESDIR

for STR in $@; do
  MATCH=(*${STR}*)
  RUNSCR=$MATCH/build/hw_runscript.sh 
  if [[ -f $RUNSCR ]]; then
    PFILE="power_trace_${STR}_${RANDOM}.csv"
    echo "Starting psamp: ./psamp.x -p 1 -p 10 -i 100 -f $PFILE"
    ssh cubert -f -C "./psamp.x -p 1 -p 10 -i 100 -f $PFILE" 
    echo "Executing $MATCH"
    ./$RUNSCR | tee $RESDIR/$STR.log
    echo "Finished $MATCH"
    ssh cubert -C "pkill psamp.x"
    scp cubert:$PFILE $RESDIR
  else
    echo "Skipping $MATCH - runscript $MATCH/hw_runscript.sh not found"
  fi
done


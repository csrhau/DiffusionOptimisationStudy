#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "$0: A Multiple Test Runner"
  echo "    Usage: $0 string1 [string2...]"
  echo "    Example: $0 1 02_loop"
  exit 1
fi


module load sdaccel/ku3-dsa-hardware

for STR in $@; do
  MATCH=(*${STR}*)
  RUNSCR=$MATCH/build/hw_runscript.sh 
  if [[ -f $RUNSCR ]]; then
    echo "Executing $MATCH"
    ./$RUNSCR
    echo "Finished $MATCH"
  else
    echo "Skipping $MATCH - runscript $MATCH/hw_runscript.sh not found"
  fi
done


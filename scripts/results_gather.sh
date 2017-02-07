#!/bin/bash

BASEDIR=..
EXPERIMENTS_DIR=$BASEDIR/experiments
APPLICATION_DIR=$BASEDIR/src

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_file>"
  exit 1;
elif [[ ! $HOSTNAME =~ .*taurus.* ]]; then
  echo "Script should be run on Taurus "
  exit 1;
elif [[ ! -d $EXPERIMENTS_DIR ]]; then
  echo "Could not locate experiments directory $EXPERIMENTS_DIR"
  exit 1;
elif [[ ! -d $APPLICATION_DIR ]]; then
  echo "Could not locate application directory $APPLICATION_DIR"
  exit 1
fi;

OUTFILE=$1

HEADER="JobID,Description,Runtime,Power"
echo $HEADER > $OUTFILE


find $EXPERIMENTS_DIR -type f -name 'slurm-*.out' | while read SLURMFILE; do
  JID=$(basename $SLURMFILE | sed -r 's/^slurm-([0-9]+).out$/\1/')
  HDEEM_FILE=$(find $APPLICATION_DIR -type f -name "deqn_sweep_*_${JID}_hdeem.csv")
  echo "Processing job $JID"
  if [[ ! -f $HDEEM_FILE ]]; then 
    echo "HDEEM File missing for $SLURMFILE, SKIPPING!"
  else
    DESCRIPTION=$(echo $SLURMFILE | cut -f 4 -d '/')
    SIMTIME=$(grep "Time Elapsed (simulation).*s" $SLURMFILE | cut -f 4 -d ' ' | sed 's/s$//g')
    SIMPOWER=$(./heartpower.py -if $HDEEM_FILE)
    if [[ -z "${SIMPOWER// }" ]]; then
      echo "Zero-length power string for JID $JID Hdeem File $HDEEM_FILE"
    else
      echo "$JID,$DESCRIPTION,$SIMTIME,$SIMPOWER"  >> $OUTFILE
    fi
  fi
done

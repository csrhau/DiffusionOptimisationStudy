#!/bin/bash

for i in {1..5}; do 
  ./run_instrumented.sh 06 11 13 03 12 05 08 02 07 04 01 09 10
  ssh cubert -C 'rm power_trace*.csv'
done

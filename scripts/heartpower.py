#!/usr/bin/env python3

import argparse
import csv
import pandas as pd

def parse_arguments():
  ''' Command Line Argument Parsing '''
  parser = argparse.ArgumentParser(description="POSE Summary Utility",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-if", "--infile", type=argparse.FileType('r'), required=True,
                      help="Path to hdeem data file to process")
  return parser.parse_args()


def hdeem_extract(textfile):
  ''' Read HDEEM data from HDEEM output CSV '''
  data = []
  in_blade = False
  in_vr = False
  for line in textfile:
    if len(line.strip()) == 0:
      if data and (in_blade or in_vr): 
        yield data
      in_blade = False
      in_vr = False
      data = []
    elif 'BLADE' in line:
      in_blade = True
      in_vr = False
    elif 'CPU0' in line:
      in_blade = False 
      in_vr = True 
    elif in_blade or in_vr:
      data.append("".join(line.strip().split()).rstrip(","))

def main():
  ''' Application Entry Point '''
  args = parse_arguments()
  blade_data, vr_data = hdeem_extract(args.infile)
  args.infile.close()
  blade = list(csv.reader(blade_data))
  vr = list(csv.reader(vr_data))
  blade_df = pd.DataFrame.from_records(blade, columns = ['Sample','Blade'], 
                                       index='Sample').apply(pd.to_numeric)
  blade_df = blade_df[6000:-3000]
  if len(blade_df) < 20000:
    print("Less than 20 seconds of core runtime!")
  else:
    print(format(blade_df.mean()[0]))
if __name__ == '__main__':
  main()

import sys
import re
import argparse 
from parselogs import parselogs, Measure
from writedata import writedata, Value
import pandas as pd
from add_data import *

parser = argparse.ArgumentParser()
parser.add_argument('--transpose', action='store_true')
parser.add_argument('--stdout', action='store_true')
parser.add_argument('--output', action='store', default='../data/output.csv')
parser.add_argument('input_file', nargs='?', default='../data/source.txt')

if __name__ == '__main__':
    args = parser.parse_args()
    logs = parselogs(args.input_file)
    if args.transpose:
        values = [Value(m.city, m.date, m.count) for m in logs]
    else:
        values = [Value(m.date, m.city, m.count) for m in logs]
    writedata(values, args.output)
    
    df_regions = read_regions_data('../data/regions_data.xlsx')
    add_data_to_csv(df_regions, args.output,  '../data/regions_data.xlsx')

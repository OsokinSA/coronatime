# covid-adapter
Parser for https://t.me/COVID2019_official region data

# Usage
python3 parse.py [--transpose] [--output OUTPUT] input_file

positional arguments:
  input_file - default data/source.txt

optional arguments:
  --transpose - changes columns to regions and rows to dates
  --output OUTPUT - output file, default data/output.csv

# Data
Ready to use parse results are committed as tab delimited CSV:
- [columns-dates](data/output.csv)
- [columns-dates](data/output-transposed.csv)
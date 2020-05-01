import re
from datetime import datetime

class Measure:
    def __init__(self, city, date, count):
        self.city = city
        self.date = date
        self.count = count
    def __eq__(self, other):
        if isinstance(other, Measure):
            return self.city == other.city \
                    and self.date == other.date \
                    and self.count == other.count
        return False

dateFormat = re.compile(r'\d\d\.\d\d\.\d\d')
regionFormat = re.compile(r'🔸.*-')
regionCountExtractor = re.compile(r'\d+\s*$')


def is_region_line(line):
    return regionFormat.search(line)

def is_date_line(line):
    return dateFormat.search(line)

def get_date(line):
    mdy = dateFormat.findall(line)[0]
    date = datetime.strptime(mdy, '%d.%m.%y').date()
    return date.isoformat()

def get_region_count(line):
    # tokens = line.split()
    region = regionFormat.findall(line)[0][1:-1].strip()
    count = int(regionCountExtractor.findall(line)[0])
    return (region, count)  

def parse_lines(lines):
    parse_result = []
    current_date = '1970-01-01'
    for line in lines:
        if is_date_line(line):
            current_date = get_date(line)
        elif is_region_line(line):
            (region, count) = get_region_count(line)
            parse_result.append(Measure(region, current_date, count))
    return parse_result

def parselogs(name):
    file = open(name, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return parse_lines(lines)
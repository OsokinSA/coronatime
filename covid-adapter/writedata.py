import csv

class Value:
    def __init__(self, keyA, keyB, count):
        self.keyA = keyA
        self.keyB = keyB
        self.count = count
    def __eq__(self, other):
        if isinstance(other, Value):
            return self.keyA == other.keyA \
                    and self.keyB == other.keyB \
                    and self.count == other.count
        return False

def get_column_keys(data):
    values = list(set([v.keyA for v in data]))
    values.sort()
    return values

def get_row_keys(data):
    values = list(set([v.keyB for v in data]))
    values.sort()
    return values

def get_row(data, column_keys):
    indexed = {}
    for value in data:
        indexed[value.keyA] = value.count
    
    row = [indexed[key] if key in indexed else 0 for key in column_keys]

    return row

def get_row_with_header(columns, first_column):
    return [first_column] + columns


def get_rows(data):
    rows = []
    column_keys = get_column_keys(data)
    row_keys = get_row_keys(data)

    rows.append(get_row_with_header(column_keys, ''))

    for row_key in row_keys:
        row = get_row([v for v in data if v.keyB == row_key], column_keys)
        rows.append(get_row_with_header(row, row_key))
    return rows

def writedata(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerows(get_rows(data))



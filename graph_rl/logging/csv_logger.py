import time
import csv
import json

class CSVLogger:

    def __init__(self, file_path, fieldnames, append = False):
        self.t_start = time.time()
        if append:
            self.file = open(file_path, "a")
        else:
            self.file = open(file_path, "w")
            self.file.write("#{}\n".format(json.dumps({"t_start": self.t_start})))
        self.csv_logger = csv.DictWriter(self.file, fieldnames = fieldnames)
        if not append: 
            self.csv_logger.writeheader()
        self.file.flush()

    def log(self, row_dict):
         self.csv_logger.writerow(row_dict)
         self.file.flush()

    def time_passed(self):
        return round(time.time() - self.t_start, 6)

def read_csv_log(file_path):
    columns = []
    column_dict = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0][0] != "#":
                if line_count == 0:
                    for column_name in row:
                        column = []
                        column_dict[column_name] = column
                        columns.append(column)
                else:
                    for value, column in zip(row, columns):
                        column.append(float(value))
                line_count += 1
    return column_dict


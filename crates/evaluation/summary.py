from glob import glob
import csv
import pathlib
from prettytable import PrettyTable

pt = PrettyTable()
pt.field_names = ["model", "MOTA", "HOTA", "IDF1"]

for file in glob("TrackEval/data/trackers/mot_challenge/MOT17-train/*/*summary.txt"):
    path = pathlib.Path(file)

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # skip header
        next(reader, None)
        for line in reader:
            pt.add_row([path.parent, float(line[12]), float(line[0]), float(line[29])])

print(pt)

import csv

stj = "initial state evloves to a fixed uniform state"

r = csv.reader(open('420_lab1_MasterExperiment.csv')) # Here your csv file
lines = list(r)

for i in range(906):
    if lines[8][i+1]:
        lines[8][i+1] = stj

writer = csv.writer(open('420_lab1_MasterExperiment.csv', 'w'))
writer.writerows(lines)
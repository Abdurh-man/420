from tkinter.tix import COLUMN
import matplotlib.pyplot as plt
import numpy as np 
import csv
import statistics

  
lam = []
lam_t = []
H = []
H_t = []

with open('420_lab1_MasterExperiment.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines: 
        if row[3].isnumeric():
            lam.append(int(row[3]))
        if row[4].isnumeric():
            lam_t.append(int(row[4]))
        if row[5].isnumeric():
            H.append(int(row[5]))
        if row[6].isnumeric():
            H_t.append(int(row[6]))        

print("Standard Deviation of Lambda   % s " %(statistics.stdev(lam)))
print("Standard Deviation of Lambda_T % s " %(statistics.stdev(lam_t)))
print("Standard Deviation of H        % s " %(statistics.stdev(H)))
print("Standard Deviation of H_T      % s " %(statistics.stdev(H_t)))
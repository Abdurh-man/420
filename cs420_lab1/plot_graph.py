from tkinter.tix import COLUMN
import matplotlib.pyplot as plt
import numpy as np 
import csv
  
b = []
c = []

with open('420_lab1_MasterExperiment.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:     
        if row[13] == "" or row[6] == "":
            continue
        b.append(float(row[13])) # classe
        c.append(float(row[6])) # lambda 

for i in range(49):
    y = []
    x = []
    if i == 0:
        continue

    for j in range(12):
        if i == 1:
            mult = 0
        else: 
            mult = 1

        x.append(c[(i*12+j)+(6*mult)])
        y.append(b[(i*12+j)+(6*mult)])

    plt.plot(x, y, linestyle = ' ',
        marker = 'o')
  
plt.title('Class vs H_t', fontsize = 20)
plt.ylabel("Class Behavior")
plt.xlabel("H_t")
plt.legend()
plt.show()
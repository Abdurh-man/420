import os
import threading
from itertools import product
import multiprocessing
from itertools import product

num_particles = []
inertia = []
cognition = []
social = []
run = []

for i in [int((j) / 1) for j in range(10, 110, 10)]:
    num_particles.append(i)
for i in [float(j) / 10 for j in range(1, 11, 1)]:
    inertia.append(i)
for i in [float(j) / 10 for j in range(1, 41, 1)]:
    cognition.append(i)
for i in [float(j) / 10 for j in range(1, 41, 1)]:
    social.append(i)
for i in range(20):
    run.append(i)

def exec_command(type,i,it):
   os.system("python pso.py --{} {} > results_Booth/{}/{}_{}_{}.txt".format(type,i,type,type,i,it))
   #print(type,i,it)

type_1 = ['num_particles']
type_2 = ['inertia']
type_3 = ['cognition']
type_4 = ['social']

if __name__ == "__main__":
    # with multiprocessing.Pool(processes=5) as pool:

    with multiprocessing.Pool() as pool:
        pool.starmap(exec_command, product(type_1,num_particles,run))
        pool.starmap(exec_command, product(type_2,inertia,run))
        pool.starmap(exec_command, product(type_3,cognition,run))
        pool.starmap(exec_command, product(type_4,social,run))       

import os
import time

start_time = time.time()

for i in [int((j) / 1) for j in range(10, 110, 10)]:
    for it in range(4):
        os.system("python pso.py --num_particles {} > results/Particles_{}_{}.txt".format(i,i,it))

for i in [float(j) / 10 for j in range(1, 11, 1)]:
    for it in range(4):
        os.system("python pso.py --inertia {} > results/Inertia_{}_{}.txt".format(i,i,it))

for i in [float(j) / 10 for j in range(1, 41, 1)]:
    for it in range(4):
        os.system("python pso.py --cognition {} > results/Cognition_{}_{}.txt".format(i,i,it))

for i in [float(j) / 10 for j in range(1, 41, 1)]:
    for it in range(4):
        os.system("python pso.py --social {} > results/Social_{}_{}.txt".format(i,i,it))


print("My program took", time.time() - start_time, "to run")
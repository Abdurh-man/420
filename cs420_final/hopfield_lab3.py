from operator import ne
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lab 3: Hopfield Networks")
    parser.add_argument("--n", default=100, help="number of neurons ", type=int)
    parser.add_argument("--p", default=50, help="patterns", type=int)

    args = parser.parse_args()    
    neurons = args.n
    pattern = args.p
    
# Define the sign
def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1
#S = []
unstable_graph = []
stable_graph = []

for run in range(5):
    # #set all the elments of counter to 0 and increment each one when 
    # #neurons are stable for their perspective p
    counter = []
    for i in range(pattern):
        counter.append(0)

    # #generating bipolar vectors
    bipolar = []

    bipolar= np.random.choice([-1,1], (pattern,neurons))
    for p in range(pattern):
        #delclare an empty weight matrix of neurons * neuerons
        weight = []
        weight = np.zeros((neurons,neurons))
        
        for i in range(neurons):
            for j in range(neurons):
                sigma = 0
                for k in range(p):
                    si = bipolar[k,i]
                    sj = bipolar[k,j]
                    sigma = sigma + (si*sj)
                weight[i,j] = (sigma/neurons)
        # prevent self coupling
        for i in range(neurons):
            weight[i,i] = 0

        for i in range(neurons):
            for j in range(i,neurons):
                weight[j,i] = weight[i,j]

        #check for stability        
        for k in range(p):
            network = []
            for i in range(neurons):
                sigma = 0.0
                for j in range(neurons):
                    sigma += weight[i,j] * bipolar[k,j]
                
                network.append(sign(sigma))

            if np.array_equal(network, bipolar[k]):
                    counter[p] = counter[p] + 1
    stable_graph.append(counter)
#set the x axis    
x = []
for p in range(pattern):
    x.append(p) 

for run in range(5):
    #plt.plot((1/big_graph[run]),y, linestyle = ' ', marker = 'o')
    plt.plot(x,stable_graph[run], linestyle = '-', marker = ' ')

plt.title('Stable Imprints N = {}, P = {} '.format(neurons,pattern), fontsize = 20)
plt.ylabel("Stable Imprints")
plt.xlabel("Number of Imprints")
plt.savefig("stable_{}_{}.jpg".format(neurons,pattern))

# clear the diagram
plt.clf()

# find the unstable fracion
for run in range(5):
    un = stable_graph[run]
    for p in range(pattern):
        un[p] = 1-(un[p]/(p+1))
    #plt.plot((1/big_graph[run]),y, linestyle = ' ', marker = 'o')
    plt.plot(x, un, linestyle = '-', marker = ' ')
plt.title('unstable Imprints N = {}, P = {} '.format(neurons,pattern), fontsize = 20)
plt.ylabel("unstable Imprints")
plt.xlabel("Number of Imprints")
plt.savefig("unstable_{}_{}.jpg".format(neurons,pattern))
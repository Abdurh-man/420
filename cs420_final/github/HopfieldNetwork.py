from pickle import TRUE
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        num_data =  len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights
        weight = np.zeros((self.num_neuron, self.num_neuron))
        #rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for t in train_data:
            weight += np.outer(t, t) / self.num_neuron
        
        # Make diagonal element of W into 0
        for i in range(self.num_neuron):
            weight[i,i] = 0
        for i in range(self.num_neuron):
            for j in range(i,self.num_neuron):
                weight[j,i] = weight[i,j]
       
        weight /= num_data 
        self.weight = weight 
    
    def predict(self, data, num_iter=128, threshold=0, asyn=TRUE):
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        # Compute initial state energy
        s = init_s
        e = self.energy(s)
        
        # Iteration
        for i in range(self.num_iter):
            # Update s
            s = np.sign(self.weight @ s - self.threshold)
            # Compute new state energy
            e_new = self.energy(s)
            
            # s is converged
            if e == e_new:
                return s
            # Update energy
            e = e_new
        return s
    
    def energy(self, s):
        return -0.5 * s @ self.weight @ s + np.sum(s * self.threshold)

# March 2022
import argparse
import os
import sys
from cgi import test

import matplotlib.pyplot as plt
import numpy as np
from distributed import Client
from leap_ec import Individual, Representation, ops, probe, test_env_var, util
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.problems import ScalarProblem
from leap_ec.decoder import IdentityDecoder
from leap_ec.distrib import DistributedIndividual, synchronous
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from toolz import pipe

import gym


class Network:
    # The network constructor takes as input an array where
    # layers[0] is the number of neurons in the first (input) layer
    # layers[1] is the number of neurons in the hidden layer
    # layers[2] is the number of neurons in the output layer
    #dtype=np.float16

    def __init__(self, layers):
        self.layers = layers
    # y_i = max(A_i, 0)
    # This function will take as input a list of the weight values and
    # setup the weights in the network based on that list
    def set_weights(self):
        weights = np.loadtxt("420_lab_4_max_fit.txt")
        self.matrix1 = np.zeros((self.layers[0],self.layers[1]))
        self.matrix2 = np.zeros((self.layers[1],self.layers[2]))

        index = 0
        for i in range(self.layers[0]):
            for j in range(self.layers[1]):
                self.matrix1[i][j] = weights[index]
                index+= 1

        for i in range(self.layers[1]):
            for j in range(self.layers[2]):
                self.matrix2[i][j] = weights[index]
                index+= 1


    # This network will take as input the observation and it will
    # calculate the forward pass of the network with that input value
    # It should return the output vector
    def forward_pass(self, obs):
        #print(self.matrix1);
        Ai = np.zeros(self.layers[1])
        #A_matrix1 = []
        #A_matrix2 = []
        #we do it twice becaese we do it between the hiden and input and the hidden and the output
        for i in range(self.layers[1]):
            sigma = 0
            for j in range(self.layers[0]):
                sigma += self.matrix1[j][i]*obs[j]
            Ai[i] = max((sigma,0))

        output_vector = np.zeros(self.layers[2])
        for i in range(self.layers[2]):
            sigma = 0
            for j in range(self.layers[1]):
                sigma += self.matrix2[j][i]*Ai[j]

            output_vector[i] = sigma

        return output_vector

# Implementation of a custom problem
class OpenAIGymProblem(ScalarProblem):
    def __init__(self, layers, env_name):
        super().__init__(maximize=True)
        self.layers = layers
        self.env_name = env_name
        self.net = Network(layers)

    def evaluate(self, ind):
        self.net.set_weights()
        score = np.zeros(5)
        env = gym.make(self.env_name)
        for i in range(5):
            observation = env.reset()
            done = False
            accumulated_score = 0

            while done == False:
                #Pass the observation through the network to get the action. The action should
                #be the argmax of the output values.
                action = self.net.forward_pass(observation)
                observation, reward, done, info = env.step(np.argmax(action))
                accumulated_score += reward

            score[i] = (accumulated_score)

        return np.mean(score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS 420/CS 527: Neuroevolution")
    parser.add_argument("--environment", type=str, help="OpenAI Gym Environmetn")
    parser.add_argument("--inputs", type=int, help="Number of inputs")
    parser.add_argument("--hidden", type=int, default= 10, help="Number of hidden")
    parser.add_argument("--outputs", type=int, help="Number of outputs")
    parser.add_argument("--trn_size", type=int, default=5, help="Tournament size")
    args = parser.parse_args()
    max_generation = 50
    N = 100
    num_inputs = args.inputs
    num_actions = args.outputs
    num_hidden = args.hidden
    layers = [num_inputs, num_hidden, num_actions]

    # Calculate the total length of the genome
    total_weights = 0
    for i in range(len(layers)-1):
        total_weights += layers[i]*layers[i+1]
    # Spin up Dask for distributed evaluation

    hidden_neurons = [10,20,30,40,50]
    #50 is for the numbe of test ran 200 is for the size of genome
    traning = []
    testing = []
    # 5 * 10 * 5
    matrx_trn = np.zeros(3,250)
    matrix_trn_index = 0
    for hid in hidden_neurons:
        ten_mean = []
        for j in range(10):
            # call the evolve_network with the correct number of hidden neurons for 10 runs
            os.system("python evolve_network.py --environment {} --inputs {} --hidden {} --outputs {}".format(args.environment, args.inputs, hid, args.outputs))
            traning.append(np.loadtxt("420_lab4_max_fit.out"))
            with Client() as client:

                # Set up the parents
                parents = DistributedIndividual.create_population(N,

                        initialize=create_real_vector(bounds=([[-1, 1]]*total_weights)),
                        decoder=IdentityDecoder(),
                        problem=OpenAIGymProblem(layers,
                            args.environment))
                        # Calculate initial fitness values for the parents
                parents = synchronous.eval_population(parents, client=client)
                # Loop over generations
                
                
                trn_size = [5, 2, 10, 20]
                for i in range(4):
                    trn_5 = []
                    for j in range(5):
                        for current_generation in range(max_generation):
                            offspring = pipe(parents,
                                    ops.tournament_selection(trn_size[i]),
                                    ops.clone,
                                    mutate_gaussian(std=0.05, hard_bounds=(-1, 1),
                                        expected_num_mutations=int(0.01*total_weights)),
                                    ops.uniform_crossover,
                                    synchronous.eval_pool(client=client, size=len(parents)))
                            fitnesses = [net.fitness for net in offspring]
                            print("Generation ", current_generation, "Max Fitness ",
                                    max(fitnesses))
                            parents = offspring
                    # Find the best network in the final population
                    index = np.argmax(fitnesses)
                    best_net = parents[index]
                    # meaning trn is 5
                    best_net.fitness
                    if i == 0:
                        trn_5.append(best_net.fit)
                    else:
                        matrx_trn[i-1][(matrix_trn_index *5) + j] = best_net.fitness
                    
                testing.append(max(trn_5))
                    
            matrix_trn_index+=1
np.savetxt("420_lab_4_trn.out",matrx_trn, delimiter =", ")
np.savetxt("420_lab_4_traning.out",traning, delimiter =", ")
np.savetxt("420_lab_4_testing.out",testing, delimiter =", ")

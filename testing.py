import os
import sys
from cv2 import accumulate
import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
from toolz import pipe
from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops, util
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.binary_rep.problems import ScalarProblem
from leap_ec.decoder import IdentityDecoder
from distributed import Client
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import synchronous

class Network:
    dtype=np.float16

    def __init__(self, layers):
        self.layers = layers

    def set_weights(self, weights):
        self.matrix1 = np.array((self.layer[0],self.layer[1]))
        self.matrix2 = np.array((self.layer[1],self.layer[2]))
        
        index = 0
        for i in range(self.layers[0]):
            for j in range(self.layers[1]):
                self.matrix1[i][j] = weights[index]
                index+= 1
        
        for i in range(self.layers[1]):
            for j in range(self.layers[2]):
                self.matrix2[i][j] = weights[index]
                index+= 1

    def forward_pass(self, obs):
        Ai = []
        output_vector = []
        for i in range(self.layers[1]):
            sigma = 0
            for j in range(self.layers[0]):
                sigma += self.matrix1[i][j]*obs[j]
            Ai.append(max(sigma,0))
        
        for i in range(self.layers[2]):
            sigma = 0
            for j in range(self.layers[1]):
                sigma += self.matrix2[i][j]*Ai[j]
            output_vector.append(sigma)
        return output_vector

class OpenAIGymProblem(ScalarProblem):
    def __init__(self, layers, env_name, ind):
        super().__init__(maximize=True)
        self.layers = layers
        self.env_name = env_name
        self.net = Network(layers)    
    
    def evaluate(self, ind):
        s
        accumulated_score = 0
        env = gym.make('CartPole-v1')
        for i in range(100):
            observation = env.reset()
            done = False
            
            while done == False:
                action = self.net.forward_pass(observation)
                observation, reward, done, info = env.step(action)
                accumulated_score += reward

        return  accumulated_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS 420/CS 527: Neuroevolution")
    parser.add_argument("--environment", type=str, help="OpenAI Gym Environmetn")
    parser.add_argument("--inputs", type=int, help="Number of inputs")
    parser.add_argument("--hidden", type=int, help="Number of hidden")
    parser.add_argument("--outputs", type=int, help="Number of outputs")
    parser.add_argument("--trn_size", type=int, default=5, help="Tournament size")

    args = parser.parse_args()
    max_generation = 50
    N = 100

    num_inputs = args.inputs
    num_actions = args.outputs
    num_hidden = args.hidden
    layers = [num_inputs, num_hidden, num_actions]

    hidden_neurons = [10,20,30,40,50]

    #50 is for the numbe of test ran 200 is for the size of genome
    master_genome = np.zeros((50,500))
    for i in range(5):
        for j in range(10):
            # call the evolve_network with the correct number of hidden neurons for 10 runs
            os.system("python evolve_network.py --environment {} --inputs {} --hidden {} --outputs {}".format(args.environment, args.inputs, hidden_neurons[i], args.outputs))
            
            file = open("420_lab_4_results.out", "r")
            
            index = 0
            
            file.close()
            # delete the file where the geneome is saved 
            os.remove("420_lab_4_results.out")

    for i in range(50):


    # Calculate the total length of the genome
    total_weights = 0
    for i in range(len(layers)-1):
        total_weights += layers[i]*layers[i+1]

    parents = Individual.create_population(N,
            initialize=create_real_vector(bounds=([[-1, 1]]*total_weights)),
            decoder=IdentityDecoder(),
            problem=OpenAIGymProblem(layers, args.environment))

    # Evaluate initial population
    parents = Individual.evaluate_population(parents)
    
    trn_size = [2, 10, 20]
    for i in range(trn_size):
        for j in range(5):
            for current_generation in range(max_generation):
                offspring = pipe(parents,
                        ops.tournament_selection(k=5),
                        ops.clone,
                        mutate_gaussian(std=0.05, hard_bounds=(-1, 1), expected_num_mutations=int(0.01*total_weights)),
                        ops.uniform_crossover,
                        ops.evaluate,
                        ops.pool(size=len(parents)))

                fitnesses = [net.fitness for net in offspring]
                print("Generation ", current_generation, "Max Fitness ", max(fitnesses))
                parents = offspring

        # Find the best network in the final population
        index = np.argmax(fitnesses)
        best_net = parents[index]

    # You may want to change how you save the best network
    print("Best network weights:")
    np.savetxt("420_lab_4_trn_size.out",best_net.genome)
    #print(best_net.genome)
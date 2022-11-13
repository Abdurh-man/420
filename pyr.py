
# CS 420/CS 527 Lab 4: Neuroevolution with LEAP
# Catherine Schuman
# March 2022
import os
import sys
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

class Network:
    # The network constructor takes as input an array where
    # layers[0] is the number of neurons in the first (input) layer
    # layers[1] is the number of neurons in the hidden layer
    # layers[2] is the number of neurons in the output layer
    dtype=np.float16

    def __init__(self, layers):
        self.layers = layers

    # This function will take as input a list of the weight values and
    # setup the weights in the network based on that list
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


    # This network will take as input the observation and it will
    # calculate the forward pass of the network with that input value
    # It should return the output vector
    def forward_pass(self, obs):
        Ai = []
        output_vector = []
        #A_matrix1 = []
        #A_matrix2 = []
        #we do it twice becaese we do it between the hiden and input and the hidden and the output
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

# Implementation of a custom problem
class OpenAIGymProblem(ScalarProblem):
    def __init__(self, layers, env_name):
        super().__init__(maximize=True)
        self.layers = layers
        self.env_name = env_name
        self.net = Network(layers)

    def evaluate(self, ind):
        score = np.array(5)
        env = gym.make('CartPole-v1')
        for i in range(5):
            observation = env.reset()
            done = False
            accumulated_score = 0

            while done == False:
                #Pass the observation through the network to get the action. The action should
                #be the argmax of the output values.
                action = self.net.forward_pass(observation)
                observation, reward, done, info = env.step(action)
                accumulated_score += reward

            score[i] = (accumulated_score)

        return np.mean(score)


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
    np.savetxt("420_lab_4_results.out",best_net.genome)
    #print(best_net.genome)
    np.arg

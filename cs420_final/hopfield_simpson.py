from operator import ne
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageOps
from matplotlib import image

import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lab 3: Hopfield Networks")
    parser.add_argument("--n", default=100, help="number of neurons ", type=int)
    parser.add_argument("--p", default=50, help="patterns", type=int)

    args = parser.parse_args()    
    neurons = args.n
    pattern = args.p

    
#~~~ gets the sign of a value
def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1

#~~~ this is used to make a bipolar vector from an image *****
img = Image.open('homer_bw_masked.png').convert('L')
img_inverted = ImageOps.invert(img)

np_img = np.array(img_inverted)

pattern = len(np_img)
neurons = len(np_img[0])

bipolar= np.ones((pattern,neurons))
for i in range(len(np_img)):
    for j in range(len(np_img[i])):
        if(np_img[i][j]):
            bipolar[i][j] = -1


#set all the elments of counter to 0 and increment each one when 
#neurons are stable for their perspective p
counter = np.zeros(pattern)

#~~~ calculates the weights of the image ~~~
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


    plt.imshow(weight) # plotting by columns
    plt.savefig("results/patern{}.jpg".format(p))
import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import os

import network

# Helper Functions
# def get_corrupted_input(input, corruption_level):
#     corrupted = np.copy(input)
#     inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
#     for i, v in enumerate(input):
#         if inv[i]:
#             corrupted[i] = -1 * v
#     return corrupted

def set_bipolar_matrix(img, w=64, h=64):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    return shift

def main():
    # Load data

    import cv2
    import glob
    img_dir = "train/" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    bipolar = []
    for f1 in files:
        img = rgb2gray(cv2.imread(f1))
        bipolar.append(img)
        
    # Preprocessing
    print("Start to data preprocessing...")
    
    # bipolar = set_bipolar_matrix(data[0])

    bipolar = [set_bipolar_matrix(d) for d in bipolar]

    # for i in range(len(input_pattern)):
#     print(input_pattern[i])
    plt.imshow(bipolar[0])
    plt.show()


if __name__ == '__main__':
    main()
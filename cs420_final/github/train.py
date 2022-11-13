import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import HopfieldNetwork
import os

#Helper Functions
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    import cv2
    import glob
    img_dir = "train/" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = rgb2gray(cv2.imread(f1))
        data.append(img)
    
    img_dir = "train/" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    test = []
    for f1 in files:
        img = rgb2gray(cv2.imread(f1))
        test.append(img)

    # Preprocessing
    data = [preprocessing(d) for d in data]

    # Create Hopfield Network Model
    model = HopfieldNetwork.HopfieldNetwork()
    model.train_weights(data)

    # Generate testset
    input = [preprocessing(t) for t in test]
    test = [get_corrupted_input(i,0.70) for i in input]
    
    predicted = model.predict(test, threshold=0, asyn=False)
    
    input = [reshape(i) for i in input]

    plt.imshow(input[0])
    plt.axis('off')
    plt.savefig("hop_input{}.jpg".format(1))
    
    plt.imshow(input[1])
    plt.axis('off')
    plt.savefig("hop_input{}.jpg".format(2))
    
    plt.imshow(input[2])
    plt.axis('off')
    plt.savefig("hop_input{}.jpg".format(3))
    
    plt.imshow(input[3])
    plt.axis('off')
    plt.savefig("hop_input{}.jpg".format(4))

    test = [reshape(t) for t in test]
    plt.imshow(test[0])
    plt.axis('off')
    plt.savefig("hop_corrupted{}.jpg".format(1))
    
    plt.imshow(test[1])
    plt.axis('off')
    plt.savefig("hop_corrupted{}.jpg".format(2))
    
    plt.imshow(test[2])
    plt.axis('off')
    plt.savefig("hop_corrupted{}.jpg".format(3))
    
    plt.imshow(test[3])
    plt.axis('off')
    plt.savefig("hop_corrupted{}.jpg".format(4))

    predicted = [reshape(d) for d in predicted]

    plt.imshow(predicted[0])
    plt.axis('off')
    plt.savefig("hop_predicted{}.jpg".format(1))
    
    plt.imshow(predicted[1])
    plt.axis('off')
    plt.savefig("hop_predicted{}.jpg".format(2))
    
    plt.imshow(predicted[2])
    plt.axis('off')
    plt.savefig("hop_predicted{}.jpg".format(3))
    
    plt.imshow(predicted[3])
    plt.axis('off')
    plt.savefig("hop_predicted{}.jpg".format(4))

if __name__ == '__main__':
    main()
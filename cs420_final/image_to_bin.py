# from PIL import Image
# import numpy as np

# img = Image.open('homer_original.png').convert('L')

# np_img = np.array(img)
# np_img = ~np_img  # invert B&W
# np_img[np_img > 0] = 1

# print(np_img)

from PIL import Image, ImageOps
import numpy as np

img = Image.open('homer_bw.png').convert('L')
img_inverted = ImageOps.invert(img)

np_img = np.array(img_inverted)

row = len(np_img)
col = len(np_img[0])

bipolar= np.ones((row,col))
for i in range(len(np_img)):
    for j in range(len(np_img[i])):
        if(np_img[i][j]):
            bipolar[i][j] = -1


plot(data, test, predicted)
# bipolar= np.array((row,col))

#print(np_img)
print(row,col)
for i in range(len(bipolar)):
    string = ""
    for j in range(len(bipolar[i])):
        string += str(int(bipolar[i][j])) + " "
    #print(string)

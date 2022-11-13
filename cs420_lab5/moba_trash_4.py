# hidden = [10,20,30,40,50]

# for hid in hidden:
#     for i in range (10):
#         os.system("python evolve_network.py --environment {} --inputs {} --hidden {} --outputs {}".format("LunarLander-v2", 8, hid,4))

# c = []

# data1 = []
# data2 = []
# data3 = []
# data4 = []
# data5 = []

# file1 = open('testing_2.txt', 'r')
# for line in file1:
#     c.append(int(float(line.strip('\n'))))
# file1.close


# for i in range(10):
#     data1.append(c[i+(10*0)])
#     data2.append(c[i+(10*1)])
#     data3.append(c[i+(10*2)])
#     data4.append(c[i+(10*2)])
#     data5.append(c[i+(10*2)])


# data = [data1, data2, data3,data4,data5]

# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_subplot(111)

# # Creating axes instance
# bp = ax.boxplot(data, patch_artist = True,
#                 notch ='True', vert = 1)


# # x-axis labels
# ax.set_xticklabels(['10','20','30', '40', '50'])

# # Adding title
# plt.title("Testing Score By Hidden Neurons")

# plt.ylabel("Testing Score")
# plt.xlabel("Hidden Neurons")

# plt.savefig("testing_2.jpg")


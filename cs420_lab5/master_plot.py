import os
import matplotlib.pyplot as plt
import statistics


num_particles = []
inertia = []
cognition = []
social = []

x1 = []
x2 = []
x3 = []
x4 = []

type_1 = ['results_Booth','results_Rosenbrock']

def graph(name,x,y):
    half = int(len(y)/2)
    
    y_conv_avg_1 = []
    y_conv_avg_2 = []
    
    Booth_data = []
    Rosenbrock_data = []

    y_conv_avg_1.append(None)
    y_conv_avg_2.append(None)
    
    for i in range(half):
        if len(y[i]) != 0:
            y_conv_avg_1.append(statistics.mean(y[i]))
            Booth_data.append(y[i])
        else:
            y_conv_avg_1.append(0)
            Booth_data.append(0)
        
        if len(y[i + half]) != 0:
            y_conv_avg_2.append(statistics.mean(y[i+half]))
            Rosenbrock_data.append(y[i+half])
        else:
            y_conv_avg_2.append(0)
            Rosenbrock_data.append(0)
              

    

    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)

    plt.boxplot(Booth_data)


    plt.plot(y_conv_avg_1, linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('Booth epoches',name), fontsize = 20)
    plt.ylabel("epoches to converge")
    plt.xlabel(name)
    plt.savefig("{}_{}.jpg".format('results_Booth',name))

    plt.clf()


    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    plt.boxplot(Rosenbrock_data)

    plt.plot(y_conv_avg_2, linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('Rosenbrock epoches',name), fontsize = 20)
    plt.ylabel("epoches to converge")
    plt.xlabel(name)
    plt.savefig("{}_{}.jpg".format('results_Rosenbrock',name))
    
    plt.clf()


def non_cong_graph(name,x,y):
    half = int(len(y)/2)

    plt.plot(x[:half],y[:half])
    plt.title('{} {}'.format('non-converged Booth',name), fontsize = 20)
    plt.ylabel("number of non-converged test")
    plt.xlabel(name)
    plt.savefig("{}_{}.jpg".format('non-converged results_Booth',name))
    plt.clf()

    plt.plot(x[half:],y[half:])
    plt.title('{} {}'.format('non-converged Rosenbrock',name), fontsize = 20)
    plt.ylabel("number of non-converged test")
    plt.xlabel(name)
    plt.savefig("{}_{}.jpg".format('non-converged results_Rosenbrock',name))
    plt.clf()

def read(conv, graph, directory,file,arg):
    number_conv = 0
    array_for_converge = []
    for i in range(20):
        f = open("{}/{}/{}_{}_{}.txt".format(directory,file,file,arg,i),'r')
        data = []
        for line in f:
            #ignores \n character 
            data.append(line[:-1])
        f.close()
        flowat = float(data[6][9:])
        if 1e-10 > flowat:
            number_conv+=1
            array_for_converge.append(int(data[4][12:]))
    #print(array_for_converge)
    conv.append(20 - number_conv)
    graph.append(array_for_converge)
#num_particles: inertia: cognition: social: epoch_stop: solution_found: fitness: 
#     0            1         2         3         4             5            6

conv_particles = []
conv_inertia = []
conv_cognition = []
conv_social = []

for pso in type_1:
    for i in [int((j) / 1) for j in range(10, 110, 10)]:
        read(conv_particles,num_particles,pso,'num_particles',i)
        x1.append(i)

    for i in [float(j) / 10 for j in range(1, 11, 1)]:
        read(conv_inertia,inertia,pso,'inertia',i)
        x2.append(i)

    for i in [float(j) / 10 for j in range(1, 41, 1)]:
        read(conv_cognition,cognition,pso,'cognition',i)
        x3.append(i)

    for i in [float(j) / 10 for j in range(1, 41, 1)]:
        read(conv_social,social,pso,'social',i)
        x4.append(i)

graph('num_particles',x1,num_particles)
graph('inertia',x2,inertia)
graph('cognition',x3,cognition)
graph('social',x4,social)

non_cong_graph('num_particles',x1,conv_particles)
non_cong_graph('inertia',x2,conv_inertia)
non_cong_graph('cognition',x3,conv_cognition)
non_cong_graph('social',x4,conv_social)

print(conv_inertia)
print(x2)
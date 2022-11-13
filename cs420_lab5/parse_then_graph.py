import matplotlib.pyplot as plt

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
    x_half = int(len(x)/2)
    y_half = int(len(y)/2)
    
    plt.plot(x[:x_half],y[:y_half], linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('Booth',name), fontsize = 20)
    plt.ylabel(name)
    plt.xlabel("parameter value")
    plt.savefig("{}_{}.jpg".format('results_Booth',name))

    plt.clf()

    plt.plot(x[:x_half],y[y_half:], linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('Rosenbrock',name), fontsize = 20)
    plt.ylabel(name)
    plt.xlabel("parameter value")
    plt.savefig("{}_{}.jpg".format('results_Rosenbrock',name))
    
    plt.clf()

    for i in range(len(y)):
        y[i] = 20 - y[i]

    plt.plot(x[:x_half],y[:y_half], linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('non-converged Booth',name), fontsize = 20)
    plt.ylabel(name)
    plt.xlabel("parameter value")
    plt.savefig("{}_{}.jpg".format('non-converged results_Booth',name))

    plt.clf()

    plt.plot(x[:x_half],y[y_half:], linestyle = '-', marker = ' ')
    plt.title('{} {}'.format('non-converged Rosenbrock',name), fontsize = 20)
    plt.ylabel(name)
    plt.xlabel("parameter value")
    plt.savefig("{}_{}.jpg".format('non-converged results_Rosenbrock',name))
    
    plt.clf()

def read(graph, directory,file,arg):
    number_to_convergene = 0
    for i in range(20):
        f = open("{}/{}/{}_{}_{}.txt".format(directory,file,file,arg,i),'r')
        data = []
        for line in f:
            #ignores \n character 
            data.append(line[:-1])
        f.close()
        flowat = float(data[6][9:])
        if 1e-10 > flowat:
            number_to_convergene+=1
    graph.append(number_to_convergene)
#num_particles: inertia: cognition: social: epoch_stop: solution_found: fitness: 
#     0            1         2         3         4             5            6

for pso in type_1:
    for i in [int((j) / 1) for j in range(10, 110, 10)]:
        read(num_particles,pso,'num_particles',i)
        x1.append(i)

    for i in [float(j) / 10 for j in range(1, 11, 1)]:
        read(inertia,pso,'inertia',i)
        x2.append(i)

    for i in [float(j) / 10 for j in range(1, 41, 1)]:
        read(cognition,pso,'cognition',i)
        x3.append(i)

    for i in [float(j) / 10 for j in range(1, 41, 1)]:
        read(social,pso,'social',i)
        x4.append(i)

graph('num_particles',x1,num_particles)
graph('inertia',x2,inertia)
graph('cognition',x3,cognition)
graph('social',x4,social)
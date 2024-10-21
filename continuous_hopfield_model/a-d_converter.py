import numpy as np
from math import tanh
class Hopfield_Model:
    # initialization
    def __init__(self, num_of_neurons):
        self.num_of_neurons = num_of_neurons
        self.u0 = 0.5
        self.delta_t = 0.001
        self.energy = []
        # set T
        self.T = np.zeros((self.num_of_neurons,self.num_of_neurons))
        for i in range(self.num_of_neurons):
            for j in range(self.num_of_neurons):
                if j != i:
                    self.T[i][j] = - 2**(i+j)
                else: self.T[i][j] = 0
        # set u, V
        self.u = np.zeros(self.num_of_neurons)
        self.u1 = self.u.copy()
        self.V = np.zeros(self.num_of_neurons)
    # set I
    def set_I(self, X=int): 
        self.X = X
        self.I = np.zeros(self.num_of_neurons)
        self.I = [- 2**(2*i - 1) + 2**i * X for i in range(self.num_of_neurons)]
    # update function
    def update(self):
        self.V = [1/2 * (1 + tanh(self.u[i]/self.u0)) for i in range(self.num_of_neurons)]
        for i in range(self.num_of_neurons):
            dot = 0
            for j in range(self.num_of_neurons):
                dot += self.T[i][j]*self.V[j]
            delta_u = self.delta_t * (self.I[i] -self.u[i] + dot)
            self.u1[i] += delta_u
        self.u = self.u1.copy()    
    # processing
    def process(self, times):
        for j in range(times):
            self.update()
            self.energy.append(self.energy_cal())
        return self.V, self.energy
    
    def energy_cal(self):
        X1 = 0
        for i in range(self.num_of_neurons):
            X1 += 2**i * self.V[i]
        energy =  1/2 * (self.X - X1)**2
        for i in range(self.num_of_neurons):
            energy = energy + 2**(2*i-1) * self.V[i] * (1-self.V[i])
        return energy
        
    
i = 13
model = Hopfield_Model(4)
model.set_I(i)
v, energy = model.process(150000)
print('X = ', i, ' v =', v, 'energy = ',energy[-1])
import matplotlib.pyplot as plt
x = np.arange(0,150000)
plt.bar(x, energy, color='blue', alpha=0.7)
plt.xlabel('Number of Iterations')
plt.ylabel('Energy')

plt.savefig('/Users/qiyuchen/Desktop/3.png')
plt.show()
    
          
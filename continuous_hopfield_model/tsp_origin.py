import numpy as np

class Hopfield:
    def __init__(self, num_of_cities, A = 100, D = 100, study_rate = 0.01, u0 = 0.2):
        self.num_of_cities = num_of_cities
        self.distance = np.zeros((self.num_of_cities, self.num_of_cities))
        self.u = np.zeros((self.num_of_cities, self.num_of_cities))
        self.V = np.zeros((self.num_of_cities, self.num_of_cities))
        self.A = A
        self.D = D
        self.u0 = u0
        self.study_rate = study_rate
        self.energy = []
        for X in range(self.num_of_cities):
            for i in range(self.num_of_cities):
                self.u[X][i] = np.arctanh(2/self.num_of_cities - 1)*(1 + np.random.uniform(-0.1,0.1))
        
    def cal_energy(self):
        energy = 0
        for X in range(self.num_of_cities):
            t = 0
            for i in range(self.num_of_cities):
                t += self.V[X][i]
            energy += 0.5 * self.A * (t-1)**2
        for i in range(self.num_of_cities):
            t = 0
            for X in range(self.num_of_cities):
                t += self.V[X][i]
            energy += 0.5 * self.A * (t-1)**2
        for X in range(self.num_of_cities):
            for Y in range(self.num_of_cities):
                for i in range(self.num_of_cities):
                    energy += 0.5 * self.D * self.distance[X][Y] * self.V[X][i%self.num_of_cities] * \
                        self.V[Y][(i+1)%self.num_of_cities]
        return energy
    
    def get_distance(self, distance):
        self.distance = distance
        
    def update(self):
        for X in range(self.num_of_cities):
            for i in range(self.num_of_cities):
                self.V[X][i] = 0.5 * (1 + np.tanh(self.u[X][i]/self.u0))

        for X in range(self.num_of_cities):
            for i in range(self.num_of_cities):
                delta = 0
                term1, term2 = 0,  0
                for j in range(self.num_of_cities):
                    term1 += self.V[X][j]
                for Y in range(self.num_of_cities):
                    term2 += self.V[Y][i]
                delta = - self.A * (term1 - 1) - self.A * (term2 - 1)
                for Y in range(self.num_of_cities):
                    delta += -self.D * self.distance[X][Y] * self.V[Y][(i+1)%self.num_of_cities] 
                self.u[X][i] += delta * self.study_rate
        
    def process(self, times):
        for i in range(1, times+1):
            self.update()
            self.energy.append(self.cal_energy())
        return self.V, self.energy
np.random.seed(3407)
points = np.random.rand(10, 2)
distance_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
tsp = Hopfield(num_of_cities=10)
tsp.get_distance(distance=distance_matrix)
import matplotlib.pyplot as plt
V, energy = tsp.process(1000)
print(V, energy)
plt.imshow(V)
plt.colorbar() 
plt.title('10 Cities')
plt.savefig('/Users/qiyuchen/Desktop/9.png')
plt.show()  
plt.clf()
x = np.arange(1,1001)
plt.bar(x, energy, color='blue', alpha=0.7)
plt.xlabel('Number of Iterations')
plt.ylabel('Energy')

plt.savefig('/Users/qiyuchen/Desktop/10.png')
plt.show()  

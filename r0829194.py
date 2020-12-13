import Reporter
import numpy as np
import random
import time
import math

# Modify the class name to match your student number.


class r0829194:
    reporter: Reporter.Reporter
    solution: list
    best: float
    population: list
    distancematrix: np.ndarray
    n_cities: int
    pop_size: int
    num_offspring: int
    tolerance: float
    checkInterval: int
    k: int

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.solution = []
        self.best = []
        self.population = []
        self.offspring = []
        self.distanceMatrix = []
        self.n_cities = 0
        self.pop_size = 0
        self.num_offspring = 0
        self.tolerance = 1e-12
        self.checkInterval = 20
        self.k = 5

    # The evolutionary algorithm's main loop
    def optimize(self, filename) -> int:
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        self.n_cities = self.distanceMatrix.shape[0]
        self.pop_size = 200
        self.num_offspring = self.pop_size
        self.k = math.ceil(self.pop_size*0.05)

        self.population = self.initialization(self.pop_size, self.n_cities)
        self.offspring = [self.Individual]*self.num_offspring

        self.population[0].distance(self.population[1])

        values = [ind.evaluate(self.distanceMatrix) for ind in self.population]
        previousMeanTourLen = sum(values)/len(values)
        previousBestTourLen = min(values)
        while True:

            # Your code here.
            iters = 500
            for i in range(iters):

                for j in range(self.num_offspring):
                    p1 = self.selection()
                    p2 = self.selection()
                    self.offspring[j] = (self.recombination(p1, p2))

                self.mutation()  # mutates whole population and offsprings

                self.population = self.elimination()
                values = [ind.evaluate(self.distanceMatrix)
                          for ind in self.population]
                meanTourLen = sum(values)/len(values)
                bestTourLen = min(values)
                bestTour = self.population[values.index(bestTourLen)].path
                bestTour = np.append(bestTour, bestTour[0])
                timeLeft = self.reporter.report(
                    meanTourLen, bestTourLen, bestTour)

                # stopping criterion
                if i % self.checkInterval == self.checkInterval - 1:
                    if abs(meanTourLen-previousMeanTourLen) < self.tolerance \
                            and abs(bestTourLen-previousBestTourLen)/abs(meanTourLen) < self.tolerance:
                        break
                    previousMeanTourLen = meanTourLen
                    previousBestTourLen = bestTourLen

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = -1
            # timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)
            if timeLeft < 0:
                break

        # Your code here.
        return 0

    # mutate all of population and offsprings

    def mutation(self) -> None:
        for ind in self.population:
            ind.try_to_mutate
        for ind in self.offspring:
            ind.try_to_mutate

    # OX recombination
    # return type as a string: https://www.python.org/dev/peps/pep-0484/#forward-references
    def recombination(self, p1, p2) -> 'self.Individual':
        a = random.randrange(0, self.n_cities-1)
        b = random.randrange(a, self.n_cities)
        offspring = [-1] * self.n_cities
        offspring[a:b] = p1.path[a:b]
        from_p1 = set(offspring[a:b])
        i = b
        for idx in range(b, self.n_cities):
            while p2.path[i] in from_p1:
                from_p1.remove(p2.path[i])
                i = 0 if i == self.n_cities-1 else i+1
            offspring[idx] = p2.path[i]
            i = 0 if i == self.n_cities-1 else i+1

        for idx in range(0, a):
            while p2.path[i] in from_p1:
                from_p1.remove(p2.path[i])
                i = 0 if i == self.n_cities-1 else i+1
            offspring[idx] = p2.path[i]
            i = 0 if i == self.n_cities-1 else i+1

        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip + beta * \
            (p2.mut_prob_flip - p1.mut_prob_flip)
        mut_prob_shuf = p1.mut_prob_shuf + beta * \
            (p2.mut_prob_shuf - p1.mut_prob_shuf)
        mut_prob_swap = p1.mut_prob_swap + beta * \
            (p2.mut_prob_swap - p1.mut_prob_swap)

        return self.Individual(np.array(offspring), max(0.01, mut_prob_flip), max(0.05, mut_prob_shuf), max(0.001, mut_prob_swap))

    # mu+lambda elimination
    def elimination(self) -> list:
        combined = self.population + self.offspring
        values = np.array([ind.evaluate(self.distanceMatrix)
                           for ind in combined])
        indices = values.argsort()[:len(self.population)]
        return [combined[i] for i in indices]

    # roullette wheel selection
    # def selection(self, weights) -> list:
    #     return random.choices(self.population, weights=weights, k=2)

    # k-tournament selection
    def selection(self) -> 'self.Individual':
        selected = random.choices(self.population, k=self.k)
        values = [ind.evaluate(self.distanceMatrix) for ind in selected]
        return selected[values.index(min(values))]

    # random initialization
    def initialization(self, pop_size, n_cities) -> list:
        def rand(): return max(0.01, 0.1 + 0.02 * np.random.uniform(-1.0, 1.0))
        def rand2(): return max(0.005, 0.05 + 0.1 * np.random.uniform(-1.0, 1.0))
        def rand3(): return max(0.002, 0.02 + 0.01 * np.random.uniform(-1.0, 1.0))
        return [self.Individual(np.random.permutation(n_cities), rand(), rand2(), rand3()) for i in range(pop_size)]

    class Individual:
        path: np.array
        mut_prob_flip: float
        mut_prob_swap: float
        mut_prob_shuf: float
        path_cost: float
        n_cities: int

        def __init__(self, path, mut_prob_flip, mut_prob_shuf, mut_prob_swap):
            self.path = path  # ndarray
            self.mut_prob_flip = mut_prob_flip
            self.mut_prob_shuf = mut_prob_shuf
            self.mut_prob_swap = mut_prob_swap
            self.path_cost = None
            self.n_cities = path.shape[0]

        def try_to_mutate(self) -> None:
            if self.mut_prob_flip > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                weights = [math.exp(math.log(0.1)/(self.n_cities-1)*(x-a))
                           for x in range(a, self.n_cities)]
                b = random.choices(range(a, self.n_cities), weights, k=1)[0]
                self.path[a:b] = np.flip(self.path[a:b])
                self.path_cost = None

            if self.mut_prob_shuf > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                weights = [math.exp(math.log(0.05)/(self.n_cities-1)*(x-a))
                           for x in range(a, self.n_cities)]
                b = random.choices(range(a, self.n_cities), weights, k=1)[0]
                np.random.shuffle(self.path[a:b])
                self.path_cost = None

            if self.mut_prob_swap > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                b = random.randrange(0, self.n_cities-1)
                self.path[a], self.path[b] = self.path[b], self.path[a]
                self.path_cost = None

        def evaluate(self, d_matrix) -> float:
            if self.path_cost:
                return self.path_cost

            s_path = 0
            for i in range(1, self.path.shape[0]):
                s_path += d_matrix[self.path[i-1]][self.path[i]]

            self.path_cost = s_path + \
                d_matrix[self.path[self.path.shape[0]-1]][self.path[0]]
            return self.path_cost

        def is_valid(self) -> bool:
            s = set(self.path)
            return len(s) == len(self.path)

        def edges(self) -> np.array:
            edges = np.empty((self.n_cities,2),int)
            for i in range(self.n_cities-2):
                edges[i][:]=self.path[i:i+2]
            edges[-1][0]=self.path[-1]
            edges[-1][1]=self.path[0]
            return edges

        def has_edge(self, edge) -> bool:
            ind = np.where(self.path == edge[0])
            if ind != self.n_cities-1:
                print(edge[1])
                return self.path[ind+1] == edge[1]
            else:
                return self.path[0] == edge[1]

        def distance(self, that) -> float:
            thatInd = np.where(that.path == self.path[0])[0][0]
            distance = 0
            for edge in self.edges():
                distance += 0 if that.has_edge(edge) else 1
                thatInd = 0 if thatInd == self.n_cities-1 else thatInd+1


if __name__ == "__main__":
    c = r0829194()
    c.optimize("tour29.csv")

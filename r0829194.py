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
        # self.solution = []
        # self.best = []
        self.population = []
        self.offspring = []
        self.distanceMatrix = []
        self.n_cities = 0
        self.pop_size = 0
        self.num_offspring = 0
        self.tolerance = 1e-12
        self.checkInterval = 50
        # self.k = 5

    # The evolutionary algorithm's main loop
    def optimize(self, filename) -> int:
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        self.n_cities = self.distanceMatrix.shape[0]  # number of cities
        self.pop_size = 200  # desired lambda size
        self.num_offspring = self.pop_size  # offspring generation size
        # self.k = math.ceil(self.pop_size*0.05) # for k-tournament

        # initialization
        self.population = self.initialization(self.pop_size, self.n_cities)
        self.offspring = [self.Individual]*self.num_offspring

        # initial cost function values
        values = [ind.evaluate(self.distanceMatrix) for ind in self.population]
        bestTourLen = sum(values)/len(values)
        meanTourLen = min(values)
        previousMeanTourLen = meanTourLen
        previousBestTourLen = bestTourLen

        i = 0
        while True:
            i += 1

            # stopping criterion
            if i % self.checkInterval == 0:
                if abs(meanTourLen-previousMeanTourLen) < self.tolerance \
                        and abs(bestTourLen-previousBestTourLen)/abs(meanTourLen) < self.tolerance:
                    break
                previousMeanTourLen = meanTourLen
                previousBestTourLen = bestTourLen

            # weights for roulette wheel selection
            total_fitness = sum([i.evaluate(self.distanceMatrix)
                                 for i in self.population])
            weights = [total_fitness - (i.evaluate(self.distanceMatrix))
                       for i in self.population]

            # generater offspring
            for j in range(self.num_offspring):
                parents = self.selection(weights)
                self.offspring[j] = (
                    self.recombination(parents[0], parents[1]))
                self.offspring[j].try_to_mutate()

            # mutates whole population and offsprings, to use with mu+lamda elimition
            # self.mutation()

            # only children
            # self.population = self.offspring
            self.population = self.elimination()

            # report iteration progress
            values = [ind.evaluate(self.distanceMatrix)
                      for ind in self.population]
            meanTourLen = sum(values)/len(values)
            bestTourLen = min(values)
            bestTour = self.population[values.index(bestTourLen)].path
            bestTour = np.append(bestTour, bestTour[0])

            # reporter -> write to file
            timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)
            if timeLeft < 0:
                break


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)
            # if timeLeft < 0:
            #     break

        # Your code here.
        return 0

    # random initialization
    def initialization(self, pop_size, n_cities) -> list:
        def rand(): return max(0.01,   0.1 + 0.02 * np.random.uniform(-1.0, 1.0))
        def rand2(): return max(0.005, 0.05 + 0.1 * np.random.uniform(-1.0, 1.0))
        def rand3(): return max(0.002, 0.02 + 0.01 * np.random.uniform(-1.0, 1.0))
        return [self.Individual(np.random.permutation(n_cities), rand(), rand2(), rand3()) for i in range(pop_size)]

    # OX recombination
    # return type as a string: https://www.python.org/dev/peps/pep-0484/#forward-references
    def recombination(self, p1, p2) -> 'self.Individual':
        # try to manage interval size (normal distribution around half of size)?
        p1_slice_len = min(self.n_cities-1, max(
            1, round(np.random.normal(0.5, 0.1)*self.n_cities)))
        a = random.randrange(0, self.n_cities-p1_slice_len)
        b = a + p1_slice_len
        # init offspring list
        # offspring = np.array([ -1 for i in range(self.n_cities) ]) # init with -1 for debugging purposes
        offspring = np.empty(self.n_cities, int)
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
                i += 1
            offspring[idx] = p2.path[i]
            i += 1

        # recombine mutation probabilities
        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip + beta * \
            (p2.mut_prob_flip - p1.mut_prob_flip)
        mut_prob_shuf = p1.mut_prob_shuf + beta * \
            (p2.mut_prob_shuf - p1.mut_prob_shuf)
        mut_prob_swap = p1.mut_prob_swap + beta * \
            (p2.mut_prob_swap - p1.mut_prob_swap)

        return self.Individual(offspring, max(0.01, mut_prob_flip), max(0.05, mut_prob_shuf), max(0.001, mut_prob_swap))

    # mu+lambda elimination
    def elimination(self) -> list:
        combined = self.population + self.offspring
        combined.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        return combined[:self.pop_size]

    # mu-lambda with crowding algorithm

    def elimination_crowd(self) -> list:
        indices = values.argsort()[:len(self.population)]
        k = 10
        for i in range(self.pop_size):
            new_generation.add(combined[indicies[i]])
            selected = random.choices(combined, k)
            dist = np.array([ind.distance(new_generation[i])
                             for ind in selected])
            combined.pop(dist.index(min(dist)))
            pass
        indices = values.argsort()[:len(self.population)]
        return [combined[i] for i in indices]

        # roullette wheel selection

    def selection(self, weights) -> list:
        return random.choices(self.population, weights=weights, k=2)

    # k-tournament selection
    # def selection(self) -> 'self.Individual':
    #     selected = random.choices(self.population, k=self.k)
    #     values = [ind.evaluate(self.distanceMatrix) for ind in selected]
    #     return selected[values.index(min(values))]

    # mutate all of population and offsprings
    def mutation(self) -> None:
        # is there a way to make it a one-liner?
        for ind in self.population:
            ind.try_to_mutate
        for ind in self.offspring:
            ind.try_to_mutate

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
            self.edges_ = None
            self.n_cities = path.shape[0]

        def try_to_mutate(self) -> None:
            if self.mut_prob_flip > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                weights = [math.exp(math.log(0.1)/(self.n_cities-1)*(x-a))
                           for x in range(a, self.n_cities)]
                b = random.choices(range(a, self.n_cities), weights, k=1)[0]
                self.path[a:b] = np.flip(self.path[a:b])
                self.path_cost = None
                self.edges_ = None

            if self.mut_prob_shuf > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                weights = [math.exp(math.log(0.05)/(self.n_cities-1)*(x-a))
                           for x in range(a, self.n_cities)]
                b = random.choices(range(a, self.n_cities), weights, k=1)[0]
                np.random.shuffle(self.path[a:b])
                self.path_cost = None
                self.edges_ = None

            if self.mut_prob_swap > np.random.uniform():
                a = random.randrange(0, self.n_cities-1)
                b = random.randrange(0, self.n_cities-1)
                self.path[a], self.path[b] = self.path[b], self.path[a]
                self.path_cost = None
                self.edges_ = None

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

        def edges(self) -> set:
            if self.edges_ == None:
                self.edges_ = set([(self.path[i], self.path[i+1])
                                   for i in range(self.n_cities-1)])
                self.edges_.add((self.path[-1], self.path[0]))
            return self.edges_

        def has_edge(self, edge) -> bool:
            return edge in self.edges()

        def distance(self, that) -> float:
            distance = len(self.edges().intersection(that.edges()))


if __name__ == "__main__":
    c = r0829194()
    c.optimize("tour29.csv")

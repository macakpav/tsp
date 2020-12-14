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

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        # self.solution = []
        # self.best = []
        self.population = []
        self.offspring = []
        self.distanceMatrix = []
        self.n_cities = 0
        self.pop_size = 100  # desired lambda size
        self.num_offspring = self.pop_size  # offspring generation size
        self.tolerance = 1e-12
        self.checkInterval = 30

        # METHODS
        self.selection = self.selection_roullete_wheel
        self.mutation = self.mutation_keep_the_best
        self.elimination = self.elimination_mu_lambda_k_crowding

        # META parameters
        self.k_tournament = 5               # k-tournament selection
        self.k_crowding = 20                # k-tournament crowding
        self.keep_mutation = 20             # number of best individuals left unmutated
        self.crowding_pop_count = 50        # number of individuals purged due to crowding
        self.init_mut_prob_flip = 0.15      # initial probability of flip mutation
        self.init_mut_prob_shuf = 0.1       # initial probability of shuffle mutation
        self.init_mut_prob_swap = 0.05      # initial probability of swap mutation
        self.min_mut_prob = 0.01            # minimal mutation probability for all

    # The evolutionary algorithm's main loop
    def optimize(self, filename) -> int:
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        self.n_cities = self.distanceMatrix.shape[0]  # number of cities
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
                if not meanTourLen == float("inf"):
                    if abs(bestTourLen-previousBestTourLen)/abs(bestTourLen) < self.tolerance:
                        if self.elimination == self.elimination_mu_lambda_k_crowding:
                            print("Switching elimination method at iteration:", i)
                            print("Best tour at iteration",
                                  i, ":", bestTourLen)
                            self.elimination = self.elimination_mu_lambda
                            self.checkInterval = 50
                        if abs(meanTourLen-previousMeanTourLen)/abs(meanTourLen) < self.tolerance:
                            break
                    previousMeanTourLen = meanTourLen
                    previousBestTourLen = bestTourLen

            # weights for roulette wheel selection
            # total_fitness = sum([i.evaluate(self.distanceMatrix)
            #                      for i in self.population])

            # selection
            selected = self.selection()

            # recombination
            self.offspring = [self.recombination(
                selected[i][0], selected[i][1]) for i in range(self.num_offspring)]
            # for j in range(self.num_offspring):
            #     p1, p2 = selected[j][:]
            #     self.offspring[j] = (self.recombination(p1, p2))
            #     self.offspring[j].try_to_mutate()

            # mutation
            self.mutation()

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
        print("Best tour length:", bestTourLen, "in", i, "iterations.")
        return 0

# -------------------------------------------------------------------------------------
# Initialization

    # random initialization
    def initialization(self, pop_size, n_cities) -> list:
        def rand1(): return max(self.min_mut_prob,
                                self.init_mut_prob_flip + 0.1 * np.random.normal(0.0, 1.0))

        def rand2(): return max(self.min_mut_prob,
                                self.init_mut_prob_shuf + 0.1 * np.random.normal(0.0, 1.0))

        def rand3(): return max(self.min_mut_prob,
                                self.init_mut_prob_swap + 0.1 * np.random.normal(0.0, 1.0))
        return [self.Individual(np.random.permutation(n_cities), rand1(), rand2(), rand3()) for i in range(pop_size)]

# -------------------------------------------------------------------------------------
# Selection operators

    # roullette wheel selection
    def selection_roullete_wheel(self) -> list:
        weights = [1/(i.evaluate(self.distanceMatrix))
                   for i in self.population]
        return [random.choices(self.population, weights=weights, k=2) for _ in range(self.num_offspring)]

    # k-tournament selection
    def selection_k_tournament(self) -> list:
        selected = random.choices(self.population, k=self.k_tournament)
        values = [ind.evaluate(self.distanceMatrix) for ind in selected]
        return selected[values.index(min(values))]

# -------------------------------------------------------------------------------------
# Recombination operators

    # OX recombination
    # return type as a string: https://www.python.org/dev/peps/pep-0484/#forward-references
    def recombination(self, p1, p2) -> 'self.Individual':

        # init offspring list
        offspring = np.empty(self.n_cities, int)

        # get slice from first parent
        a, b = normal_slice(self.n_cities, 0.5, 0.1)
        offspring[:b-a] = p1.path[a:b]
        from_p1 = set(offspring[:b-a])

        # fill the rest of offspring from second parent, starting from index b
        i = b-a
        for idx in range(b, self.n_cities):
            if p2.path[idx] in from_p1:
                from_p1.remove(p2.path[idx])
                continue
            offspring[i] = p2.path[idx]
            i += 1
        # continue from beginning until b
        for idx in range(0, b):
            if p2.path[idx] in from_p1:
                from_p1.remove(p2.path[idx])
                continue
            offspring[i] = p2.path[idx]
            i += 1

        # recombine mutation probabilities
        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip + beta * \
            (p2.mut_prob_flip - p1.mut_prob_flip)
        mut_prob_shuf = p1.mut_prob_shuf + beta * \
            (p2.mut_prob_shuf - p1.mut_prob_shuf)
        mut_prob_swap = p1.mut_prob_swap + beta * \
            (p2.mut_prob_swap - p1.mut_prob_swap)

        return self.Individual(offspring, max(self.min_mut_prob, mut_prob_flip),
                               max(self.min_mut_prob, mut_prob_shuf), max(self.min_mut_prob, mut_prob_swap))

# -------------------------------------------------------------------------------------
# Mutation operators

    # mutate all of population and offsprings
    def mutation_all(self) -> None:
        # is there a way to make it a one-liner?
        for ind in self.population:
            ind.try_to_mutate()
        for ind in self.offspring:
            ind.try_to_mutate()

    # mutate all of population exept a few of the best individuals
    def mutation_keep_the_best(self) -> None:
        self.population.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        for ind in self.population[self.keep_mutation:]:
            ind.try_to_mutate()
        for ind in self.offspring:
            ind.try_to_mutate()

    # do not mutate parents
    def mutation_offspring_only(self) -> None:
        for ind in self.offspring:
            ind.try_to_mutate()
# -------------------------------------------------------------------------------------
# Elimination operators

    # mu+lambda elimination
    def elimination_mu_lambda(self) -> list:
        combined = self.population + self.offspring
        combined.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        return combined[:self.pop_size]

    # mu-lambda elimination with k-crowding
    def elimination_mu_lambda_k_crowding(self) -> list:
        combined = self.population + self.offspring
        combined.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        # indicies = sorted(range(len(combined)), key=lambda k: combined[k].evaluate(self.distanceMatrix))

        for i in range(self.crowding_pop_count):
            combatants = random.choices(
                range(i+1, len(combined)), k=self.k_crowding)
            combined.pop(min(combatants,
                             key=lambda ind: combined[i].distance(combined[ind])))
        return combined[:self.pop_size]

# -------------------------------------------------------------------------------------
# Individual class

    class Individual:
        path: np.array
        mut_prob_flip: float
        mut_prob_swap: float
        mut_prob_shuf: float
        path_cost: float
        edges: set
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
                self.mutate_flip()
            if self.mut_prob_shuf > np.random.uniform():
                self.mutate_shuffle()
            if self.mut_prob_swap > np.random.uniform():
                self.mutate_swap()

        def mutate_flip(self) -> None:
            a, b = gamma_slice(self.n_cities, 0.5)
            self.path[a:b] = np.flip(self.path[a:b])
            self.reset()

        def mutate_shuffle(self) -> None:
            a, b = gamma_slice(self.n_cities, 0.2)
            np.random.shuffle(self.path[a:b])
            self.reset()

        def mutate_swap(self) -> None:
            a = random.randrange(0, self.n_cities-1)
            b = random.randrange(0, self.n_cities-1)
            self.path[a], self.path[b] = self.path[b], self.path[a]
            self.reset()

        def reset(self) -> None:
            self.path_cost = None
            self.edges_ = None

        def evaluate(self, d_matrix) -> float:
            if self.path_cost == None:
                self.path_cost = 0.0
                for i in range(1, self.n_cities):
                    self.path_cost += d_matrix[self.path[i-1]][self.path[i]]
                self.path_cost += d_matrix[self.path[-1]][self.path[0]]
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

        def distance(self, that) -> float:
            return len(self.edges().difference(that.edges()))

# -------------------------------------------------------------------------------------
# Utility funcitons


def gamma_slice(arr_len, center_prc):
    slice_len = min(arr_len-1,
                    max(1, round(np.random.gamma(center_prc*arr_len))))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len


def normal_slice(arr_len, center_prc, spread_prc):
    slice_len = min(
        arr_len-1, max(1, round(np.random.normal(center_prc, spread_prc)*arr_len)))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len


def uniform_slice(arr_len):
    slice_len = min(
        arr_len-1, max(1, round(np.random.uniform()*arr_len)))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len

# -------------------------------------------------------------------------------------
# main


if __name__ == "__main__":
    c = r0829194()
    c.optimize("tour29.csv")

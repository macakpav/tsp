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
        self.tolerance = 1e-12  # below this the relative norm is considered zero
        self.checkInterval = 30 # before advancing stage

        # METHODS
        self.stage = 1
        self.selection = self.selection_roullete_wheel
        self.mutation = self.mutation_discard_worse
        self.local_search = lambda self: 0#self.lopt_2opt_flip_elite
        self.elimination = self.elimination_mu_lambda_k_crowding

        # META parameters

        # selection
        self.k_tournament = int(self.pop_size*0.05)               # k-tournament selection

        # crowding
        self.k_crowding = int(self.pop_size/5)                # k-tournament crowding
        self.k_crowding_pop_count = int(self.pop_size/4)      # number of individuals purged due to crowding

        # mutation
        self.no_elites = 3                  # number of best individuals kept and improved
        self.init_mut_prob_flip = 0.35      # initial probability of flip mutation
        self.init_mut_prob_shuf = 0.25      # initial probability of shuffle mutation
        self.init_mut_prob_swap = 0.15      # initial probability of swap mutation
        self.min_mut_prob = 0.02            # minimal mutation probability for all
        self.mut_boost_treshold = 0.1       # with lower mutation average a boost will ocur
        self.mut_boost_coefficient = 0.5    # mutation gets boosted by Coef * Init_mut_prob

    def start_second_stage(self):
        self.stage=2
        print("\n\nSwitching to second stage.\n\n")
        self.elimination = self.elimination_mu_lambda
        # self.local_search = self.lopt_2opt_flip_short
        self.checkInterval = (6*self.checkInterval)/5

    # The evolutionary algorithm's main loop
    def optimize(self, filename) -> int:
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        self.n_cities = self.distanceMatrix.shape[0]  # number of cities
        self.checkInterval = int(max(20,min(35,-20./1000.*self.n_cities+37.)))

        # initialization
        self.population = self.initialization(self.pop_size, self.n_cities)
        self.lopt_2opt_flip_short()
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

            mut_rate_avg = sum([(ind.mut_prob_flip_ + ind.mut_prob_shuf_ +
                                 ind.mut_prob_swap_)/3 for ind in self.population])/self.pop_size
            if mut_rate_avg < self.mut_boost_treshold:
                self.boost_mutation_rate()

            # stopping criterion
            # print(i,"\tBest:", bestTourLen, ", Mean:", meanTourLen)
            if i % self.checkInterval == 0:
                print(i,"\tBest:", bestTourLen, ", Mean:", meanTourLen
                       , "Average mutation rate: ", mut_rate_avg)
                # if not meanTourLen == float("inf"):
                if abs(bestTourLen-previousBestTourLen)/abs(bestTourLen) < self.tolerance:
                    if self.stage==1:
                        self.start_second_stage()
                    if abs(meanTourLen-previousMeanTourLen)/abs(meanTourLen) < self.tolerance:
                        self.no_elites = 2
                        self.lopt_2opt_flip_elite()
                        break #converged
                previousMeanTourLen = meanTourLen
                previousBestTourLen = bestTourLen

            # selection
            selected = self.selection()

            # recombination
            self.offspring = [self.recombination(
                selected[i][0], selected[i][1]) for i in range(self.num_offspring)]

            # mutation
            self.mutation()
            
            # local optimization
            # if self.stage>1:
            # self.local_search()

            # elimination
            self.population = self.elimination()

            # report progress
            values = [ind.evaluate(self.distanceMatrix)
                      for ind in self.population]
            meanTourLen = sum(values)/len(values)
            bestTourLen = min(values)
            bestTour = self.population[values.index(bestTourLen)].path_
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
        offspring[:b-a] = p1.path_[a:b]
        from_p1 = set(offspring[:b-a])

        # fill the rest of offspring from second parent, starting from index b
        i = b-a
        for idx in range(b, self.n_cities):
            if p2.path_[idx] in from_p1:
                from_p1.remove(p2.path_[idx])
                continue
            offspring[i] = p2.path_[idx]
            i += 1
        # continue from beginning until b
        for idx in range(0, b):
            if p2.path_[idx] in from_p1:
                from_p1.remove(p2.path_[idx])
                continue
            offspring[i] = p2.path_[idx]
            i += 1

        # recombine mutation probabilities
        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip_ + beta * \
            (p2.mut_prob_flip_ - p1.mut_prob_flip_)
        mut_prob_shuf = p1.mut_prob_shuf_ + beta * \
            (p2.mut_prob_shuf_ - p1.mut_prob_shuf_)
        mut_prob_swap = p1.mut_prob_swap_ + beta * \
            (p2.mut_prob_swap_ - p1.mut_prob_swap_)

        return self.Individual(offspring, max(self.min_mut_prob, mut_prob_flip),
                               max(self.min_mut_prob, mut_prob_shuf), max(self.min_mut_prob, mut_prob_swap))

# -------------------------------------------------------------------------------------
# Mutation operators

    # mutate all of population and offsprings
    def mutation_all(self) -> None:
        # is there a way to make it a one-liner?
        for ind in self.population + self.offspring:
            ind.try_to_mutate()

    # mutate all of population exept a few of the best individuals in parents
    def mutation_only_plebs(self) -> None:
        self.population.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        for ind in self.population[self.no_elites:] + self.offspring:
            ind.try_to_mutate()

    def mutation_discard_worse(self) -> None:
        old_path=np.empty(self.n_cities,int)
        for ind in self.population + self.offspring:
            old_val=ind.evaluate(self.distanceMatrix)
            np.copyto(old_path,ind.path_)
            ind.mutate_flip()
            if old_val < ind.evaluate(self.distanceMatrix):
                np.copyto(ind.path_, old_path)
                ind.reset()
                # debug check
                # if old_val != ind.evaluate(self.distanceMatrix):
                #     print("ERROR maybe!")
                #     exit(1)
                ind.path_cost_=old_val

    # do not mutate parents
    def mutation_only_offspring(self) -> None:
        for ind in self.offspring:
            ind.try_to_mutate()

    # increase the mutation rate of individuals in population
    def boost_mutation_rate(self) -> None:
        for ind in self.population:
            ind.mut_prob_flip_ += self.init_mut_prob_flip * self.mut_boost_coefficient
            ind.mut_prob_shuf_ += self.init_mut_prob_shuf * self.mut_boost_coefficient
            ind.mut_prob_swap_ += self.init_mut_prob_swap * self.mut_boost_coefficient


# -------------------------------------------------------------------------------------
# Local search operators
    # 2-opt using flip mutation on few elite candidates
    def lopt_2opt_flip_elite(self) -> None:
        self.population.sort(key=lambda ind: ind.evaluate(self.distanceMatrix))
        for ind in self.population[:self.no_elites]:
            self.flip_2opt(ind)

    # 2-opt using flip mutation on whole population
    def lopt_2opt_flip_all(self) -> None:
        for ind in self.population:
            self.flip_2opt(ind)

    def lopt_2opt_flip_short(self) -> None:
        pass

    # 2-opt flip operator on single individual
    def flip_2opt_fulltime(self, ind) -> None:
        best_val = ind.evaluate(self.distanceMatrix)
        best_route = np.copy(ind.path_)
        improved = True
        while (improved):
            improved = False
            for a in range(self.n_cities):
                for b in range(a+2, min(a+5,self.n_cities+1)):
                    ind.path_[a:b] = np.flip(ind.path_[a:b])
                    ind.reset()
                    if best_val > ind.evaluate(self.distanceMatrix):
                        best_val = ind.evaluate(self.distanceMatrix)
                        best_route = np.copy(ind.path_)
                        improved = True
                        break
                if improved:
                    break
            ind.path_ = best_route
            # debug check
            # if best_val != ind.evaluate(self.distanceMatrix):
            #     print("ERROR maybe!")
            #     exit(1)
            ind.path_cost_ = best_val

    # 2-opt flip operator on single individual
    def flip_2opt(self, ind) -> None:
        if ind.failed_fast_opt:
            return
        old_val = ind.evaluate(self.distanceMatrix)
        for a in range(self.n_cities):
            for b in range(a+2, min(a+3,self.n_cities+1)):
                ind.path_[a:b] = np.flip(ind.path_[a:b])
                ind.path_cost_=None
                if old_val > ind.evaluate(self.distanceMatrix):
                    ind.edges_=None
                    return
                else:
                    ind.path_[a:b] = np.flip(ind.path_[a:b])
                    ind.path_cost_=old_val
                    ind.failed_fast_opt=True


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

        for i in range(self.k_crowding_pop_count):
            combatants = random.choices(
                range(i+1, len(combined)), k=self.k_crowding)
            combined.pop(min(combatants,
                             key=lambda ind: combined[i].distance(combined[ind])))
        return combined[:self.pop_size]

# -------------------------------------------------------------------------------------
# Individual class

    class Individual:
        path_: np.array
        mut_prob_flip_: float
        mut_prob_swap_: float
        mut_prob_shuf_: float
        path_cost_: float
        edges: set
        n_cities_: int
        failed_fast_opt: bool

        def __init__(self, path, mut_prob_flip, mut_prob_shuf, mut_prob_swap):
            self.path_ = path  # ndarray
            self.mut_prob_flip_ = mut_prob_flip
            self.mut_prob_shuf_ = mut_prob_shuf
            self.mut_prob_swap_ = mut_prob_swap
            self.path_cost_ = None
            self.edges_ = None
            self.n_cities_ = self.path_.shape[0]
            self.failed_fast_opt = False

        def try_to_mutate(self) -> None:
            if self.mut_prob_flip_ > np.random.uniform():
                self.mutate_flip()
            if self.mut_prob_shuf_ > np.random.uniform():
                self.mutate_shuffle()
            if self.mut_prob_swap_ > np.random.uniform():
                self.mutate_swap()

        def mutate_flip(self) -> None:
            a, b = gamma_slice(self.n_cities_, 0.5)
            self.path_[a:b] = np.flip(self.path_[a:b])
            self.reset()

        def mutate_shuffle(self) -> None:
            a, b = gamma_slice(self.n_cities_, 0.2)
            np.random.shuffle(self.path_[a:b])
            self.reset()

        def mutate_swap(self) -> None:
            a = random.randrange(0, self.n_cities_-1)
            b = random.randrange(0, self.n_cities_-1)
            self.path_[a], self.path_[b] = self.path_[b], self.path_[a]
            self.reset()

        def reset(self) -> None:
            self.path_cost_ = None
            self.edges_ = None
            self.failed_fast_opt = False

        def evaluate(self, d_matrix) -> float:
            if self.path_cost_ == None:
                self.path_cost_ = 0.0
                for i in range(1, self.n_cities_):
                    self.path_cost_ += d_matrix[self.path_[i-1]][self.path_[i]]
                self.path_cost_ += d_matrix[self.path_[-1]][self.path_[0]]
            return self.path_cost_

        def is_valid(self) -> bool:
            s = set(self.path_)
            return len(s) == len(self.path_)

        def edges(self) -> set:
            if self.edges_ == None:
                self.edges_ = set([(self.path_[i], self.path_[i+1])
                                   for i in range(self.n_cities_-1)])
                self.edges_.add((self.path_[-1], self.path_[0]))
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
    c.optimize("tour194.csv")

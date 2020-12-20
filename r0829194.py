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
    dist_matrix: np.ndarray
    no_cities: int
    pop_size: int
    no_offspring: int
    tolerance: float
    checkInterval: int
    interation: int

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.population = []
        self.offspring = []
        self.dist_matrix = []
        self.no_cities = 0
        self.pop_size = 200  # desired lambda size
        self.no_offspring = self.pop_size*3  # offspring generation size
        self.tolerance = 1e-2
        self.checkInterval = 10
        self.iteration = 0
        self.timing = 0
        self.c = 1

        # METHODS
        self.initialization = self.initialization_nearest_neigbor
        self.selection = self.selection_round_robin
        self.recombination = self.recombination_SCX
        self.mutation = self.mutation_all_keep_better
        self.local_search = self.lopt_fast_opt_children
        self.elimination = self.elimination_round_robin_k_crowding

        # META parameters

        # selection
        # round robin selection
        self.round_robin_q_selection = int(self.pop_size*0.25)

        # recombination
        # -

        # mutation
        self.init_mut_prob_flip = 0.4       # initial probability of flip mutation
        self.init_mut_prob_shuf = 0.25      # initial probability of shuffle mutation
        self.min_mut_prob = 0.01            # minimal mutation probability for all
        self.mut_boost_treshold = 0.1       # with lower mutation average a boost will ocur
        # mutation gets boosted by Coef * Init_mut_prob
        self.mut_boost_coefficient = 0.5

        # local optimization
        self.two_opt_subinterval_size = int(self.pop_size*0.2)

        # elimination
        self.crowding_k = int((self.pop_size+self.no_offspring)*0.1)  # k-tournament crowding
        # number of individuals poped due to crowding
        self.crowding_pop_count = min(self.pop_size, self.no_offspring)
        self.round_robin_q_elimination = int(
            self.pop_size*0.15)  # round robin elimination

    # The evolutionary algorithm's main loop

    def optimize(self, filename) -> int:
        # Read distance matrix from file.
        file = open(filename)
        self.dist_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        self.no_cities = self.dist_matrix.shape[0]  # number of cities

        # initialization
        self.initialization()
        self.lopt_two_opt_only_population(10)

        # self.validity_check()
        # self.avg_distance()
        # return

        # initial cost function values
        values = [ind.evaluate() for ind in self.population]
        meanTourLen = sum(values)/len(values)
        bestTourLen = min(values)
        prevCheckBestTourLen = bestTourLen

        while True:
            self.iteration += 1
            # self.validity_check()
            # self.avg_distance()
            # mut_rate_avg = self.avg_mutation_rate()

            # mutation rate boosting
            # mut_rate_avg = sum([(ind.mut_prob_flip_ + ind.mut_prob_shuf_ +
            #                      ind.mut_prob_swap_)/3 for ind in self.population])/self.pop_size
            # if mut_rate_avg < self.mut_boost_treshold:
            #     self.boost_mutation_rate()

            # stopping criterion
            if self.iteration % self.checkInterval == 0:
                if abs(bestTourLen-prevCheckBestTourLen)/abs(bestTourLen) < self.tolerance:
                    print("Best local search...")
                    ind = min(self.population, key=lambda ind: ind.evaluate())
                    if ind.is_two_optimal_:
                        self.c *= 1.5
                        ind.is_two_optimal_ = False
                    else:
                        self.c = 1
                    self.two_opt(
                        ind, int(self.no_cities * min(1, 0.2 * self.c)))
                    if abs(meanTourLen-bestTourLen)/abs(bestTourLen) < self.tolerance:
                        if timeLeft < 50:
                            break
                        print("Purge.")
                        self.purge_init(
                            no_inds_to_keep=int(self.pop_size*0.05))
                        self.lopt_fast_opt_population()
                        # break  # converged
                    if self.avg_mutation_rate() < self.mut_boost_treshold:
                        print("Boosting mutation rate.")
                        self.boost_mutation_rate()
                prevCheckBestTourLen = bestTourLen

            # selection
            selected = self.selection()

            # recombination
            self.offspring = [self.recombination(
                selected[i][0], selected[i][1]) for i in range(self.no_offspring)]

            # mutation
            self.mutation()

            # local optimization
            # self.local_search()

            # elimination
            self.elimination()

            # report progress
            values = [ind.evaluate() for ind in self.population]
            meanTourLen = sum(values)/len(values)
            bestTourLen = min(values)
            best_guy = self.population[values.index(bestTourLen)]
            bestTour = np.append(best_guy.path_, best_guy.path_[0])

            # reporter -> write to file
            timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)
            if timeLeft < 40:
                print("Time break.")
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

        print("Last local optimizations for the best guy.")
        values = [ind.evaluate() for ind in self.population]
        meanTourLen = sum(values)/len(values)
        bestTourLen = min(values)
        best_guy = self.population[values.index(bestTourLen)]
        bestTour = np.append(best_guy.path_, best_guy.path_[0])
        timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)

        best_guy.is_two_optimal_ = False
        self.two_opt(best_guy, int(self.no_cities*0.2))
        while (timeLeft > 0.02):
            old_val = best_guy.evaluate()
            for n in range(2, self.no_cities):
                best_guy.is_two_optimal_ = False
                meanTourLen += (best_guy.evaluate() -
                                bestTourLen)/self.no_cities
                bestTourLen = best_guy.evaluate()
                bestTour = np.append(best_guy.path_, best_guy.path_[0])
                timeLeft = self.reporter.report(
                    meanTourLen, bestTourLen, bestTour)
                self.two_opt(best_guy, n, n)
            if (old_val-bestTourLen)/old_val < 1e-12:  # cant do more with this
                break

        # reporter -> write to file
        timeLeft = self.reporter.report(meanTourLen, bestTourLen, bestTour)
        print("Best tour length:", bestTourLen,
              "in", self.iteration, "iterations.")
        return 0

# -------------------------------------------------------------------------------------
# Initialization

    # random initialization
    def initialization_random(self) -> None:
        def rand1(): return max(self.min_mut_prob,
                                self.init_mut_prob_flip + 0.1 * np.random.normal(0.0, 1.0))

        def rand2(): return max(self.min_mut_prob,
                                self.init_mut_prob_shuf + 0.1 * np.random.normal(0.0, 1.0))

        self.population = [self.Individual(np.random.permutation(self.no_cities), self.dist_matrix, rand1(), rand2())
                           for _ in range(self.pop_size)]

    def initialization_nearest_neigbor(self) -> None:
        if self.pop_size <= self.no_cities:
            starting_points = random.choices(
                range(self.no_cities), k=self.pop_size)
        else:
            starting_points = list(range(self.no_cities)) + random.choices(
                range(self.no_cities), k=self.pop_size-self.no_cities)

        def rand1(): return max(self.min_mut_prob,
                                self.init_mut_prob_flip + 0.1 * np.random.normal(0.0, 1.0))

        def rand2(): return max(self.min_mut_prob,
                                self.init_mut_prob_shuf + 0.1 * np.random.normal(0.0, 1.0))

        self.population = [self.Individual(self.fill_with_nearest_neighbor(starting_points[i]), self.dist_matrix,
                                           rand1(), rand2())
                           for i in range(self.pop_size)]

    def fill_with_nearest_neighbor(self, first_city, gamma_center=0.6) -> np.ndarray:
        ind_path = np.ones(self.no_cities, int) * -1
        ind_path[0] = first_city
        # size of randomly generated part
        random_range = int(
            (self.no_cities-1) * min(1, max(0, np.random.gamma(gamma_center, 1))))

        for i in range(1, self.no_cities-random_range):
            sorted_nbs = np.argsort(self.dist_matrix[ind_path[i-1]][:])
            # first (zero-th) is always the city itself (distance=0 in the matrix)
            j = 1
            while sorted_nbs[j] in ind_path:
                j += 1
            ind_path[i] = sorted_nbs[j]

        # fill the rest from random order
        rand_guy = np.random.permutation(self.no_cities)
        j = 0
        for i in range(self.no_cities-random_range, self.no_cities):
            while rand_guy[j] in ind_path:
                j += 1
            ind_path[i] = rand_guy[j]

        return ind_path

    def purge_init(self, no_inds_to_keep=15):
        newly_generated = self.pop_size-no_inds_to_keep
        if newly_generated > self.no_cities:
            starting_points = random.choices(
                range(self.no_cities), k=newly_generated)
        else:
            starting_points = random.sample(
                range(self.no_cities), k=newly_generated)

        def rand1(): return max(self.min_mut_prob,
                                self.init_mut_prob_flip + 0.1 * np.random.normal(0.0, 1.0))

        def rand2(): return max(self.min_mut_prob,
                                self.init_mut_prob_shuf + 0.1 * np.random.normal(0.0, 1.0))

        self.population[no_inds_to_keep:] = [self.Individual(self.fill_with_nearest_neighbor(starting_points[i], gamma_center=0.8), self.dist_matrix,
                                                             rand1(), rand2())
                                             for i in range(newly_generated)]

# -------------------------------------------------------------------------------------
# Selection operators

    # roullette wheel selection
    def selection_roullete_wheel(self) -> list:
        weights = [1/(i.evaluate(self.dist_matrix))
                   for i in self.population]
        return [random.choices(self.population, weights=weights, k=2)
                for _ in range(self.no_offspring)]

    # round robin selection
    def selection_round_robin(self) -> list:
        weights = [1+sum(int(ind.evaluate() < combatant.evaluate()) for combatant
                         in random.choices(self.population, k=self.round_robin_q_selection))
                   for ind in self.population]
        return [random.choices(self.population, weights=weights, k=2)
                for _ in range(self.no_offspring)]

# -------------------------------------------------------------------------------------
# Recombination operators

    # OX recombination
    # return type as a string: https://www.python.org/dev/peps/pep-0484/#forward-references
    def recombination_OX(self, p1, p2) -> 'self.Individual':

        # init offspring list
        offspring = np.empty(self.no_cities, int)
        a, b = normal_slice(self.no_cities, 0.5, 0.1)

        offspring[:b-a] = p1.path_[:b-a]
        from_p1 = set(offspring[:b-a])
        from_p2 = [i for i in p2.path_[b:] if i not in from_p1]
        from_p2 += [i for i in p2.path_[:b] if i not in from_p1]

        offspring[b-a:] = from_p2

        # recombine mutation probabilities
        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip_ + beta * \
            (p2.mut_prob_flip_ - p1.mut_prob_flip_)
        mut_prob_shuf = p1.mut_prob_shuf_ + beta * \
            (p2.mut_prob_shuf_ - p1.mut_prob_shuf_)

        return self.Individual(offspring, self.dist_matrix, max(self.min_mut_prob, mut_prob_flip),
                               max(self.min_mut_prob, mut_prob_shuf))

    # SCX recombination
    def recombination_SCX(self, p1, p2) -> 'self.Individual':
        offspring = np.ones(self.no_cities, int) * -1
        p1_edges_dict = dict(p1.edges())
        p2_edges_dict = dict(p2.edges())
        offspring[0] = np.random.randint(0, self.no_cities)
        off_set = set([offspring[0]])
        for i in range(1, self.no_cities):
            prev_city = offspring[i-1]
            legitimate_p1 = p1_edges_dict[prev_city]
            while legitimate_p1 in off_set:
                legitimate_p1 = p1_edges_dict[legitimate_p1]
            legitimate_p2 = p2_edges_dict[prev_city]
            while legitimate_p2 in off_set:
                legitimate_p2 = p2_edges_dict[legitimate_p2]
            if self.dist_matrix[prev_city][legitimate_p1] \
                    < self.dist_matrix[prev_city][legitimate_p2]:
                offspring[i] = legitimate_p1
            else:
                offspring[i] = legitimate_p2
            off_set.add(offspring[i])

        # recombine mutation probabilities
        beta = 2 * random.random() - 0.5
        mut_prob_flip = p1.mut_prob_flip_ + beta * \
            (p2.mut_prob_flip_ - p1.mut_prob_flip_)
        mut_prob_shuf = p1.mut_prob_shuf_ + beta * \
            (p2.mut_prob_shuf_ - p1.mut_prob_shuf_)
        return self.Individual(offspring, self.dist_matrix, max(self.min_mut_prob, mut_prob_flip),
                               max(self.min_mut_prob, mut_prob_shuf))
# -------------------------------------------------------------------------------------
# Mutation operators

    # mutate all of population and offsprings
    def mutation_all(self) -> None:
        for ind in self.population + self.offspring:
            ind.try_to_mutate()

    # mutate all of population and offsprings
    def mutation_all_keep_better(self) -> None:
        for ind in self.offspring:
            if ind.try_to_mutate():
                self.fast_opt(ind)
        for ind in self.population:
            ind.flip_keep_better()

    # do not mutate parents
    def mutation_only_offspring(self) -> None:
        for ind in self.offspring:
            ind.try_to_mutate()

    # do not mutate parents, fast-opt if mutated
    def mutation_only_offspring_fast_opt(self) -> None:
        for ind in self.offspring:
            if ind.try_to_mutate():
                self.fast_opt(ind)

    # mutate all of population but keep only improved individuals
    def mutation_discard_worse(self) -> None:
        old_path = np.empty(self.no_cities, int)
        for ind in self.population + self.offspring:
            old_val = ind.evaluate()
            np.copyto(old_path, ind.path_)
            ind.try_to_mutate()
            if old_val < ind.evaluate():
                np.copyto(ind.path_, old_path)
                ind.reset()
                ind.path_cost_ = old_val

    # increase the mutation rate of individuals in population
    def boost_mutation_rate(self) -> None:
        for ind in self.population:
            ind.mut_prob_flip_ += self.init_mut_prob_flip * self.mut_boost_coefficient
            ind.mut_prob_shuf_ += self.init_mut_prob_shuf * self.mut_boost_coefficient


# -------------------------------------------------------------------------------------
# Local search operators


# ------------
# Fast-opt

    def lopt_fast_opt_all(self) -> None:
        for ind in self.population + self.offspring:
            self.fast_opt(ind)

    def lopt_fast_opt_population(self) -> None:
        for ind in self.population:
            self.fast_opt(ind)

    def lopt_fast_opt_children(self) -> None:
        for ind in self.offspring:
            self.fast_opt(ind)

    def fast_opt(self, ind) -> None:
        for a in range(-2, self.no_cities-2):
            self.fast_opt_at(ind, a-1, a, a+1, a+2)

    def fast_opt_at(self, ind, a_1, a, b, b_1) -> None:
        old_val = self.dist_matrix[ind.path_[a_1], ind.path_[a]]\
            + self.dist_matrix[ind.path_[a], ind.path_[b]] \
            + self.dist_matrix[ind.path_[b], ind.path_[b_1]]
        new_val = self.dist_matrix[ind.path_[a_1], ind.path_[b]] \
            + self.dist_matrix[ind.path_[b], ind.path_[a]] + \
            self.dist_matrix[ind.path_[a], ind.path_[b_1]]
        if new_val < old_val:
            ind.path_[a], ind.path_[b] = ind.path_[b], ind.path_[a]
            ind.reset()

# ------------
# 2-Opt

    def lopt_two_opt_all(self, subinterval_size=5) -> None:
        for ind in self.population + self.offspring:
            self.two_opt(ind, subinterval_size)

    def lopt_two_opt_only_population(self, subinterval_size=5) -> None:
        for ind in self.population:
            self.two_opt(ind, subinterval_size)

    def lopt_two_opt_only_offspring(self, subinterval_size=5) -> None:
        for ind in self.offspring:
            self.two_opt(ind, subinterval_size)

# 2-Opt (variable size of sub-interval)
    def two_opt(self, ind, max_subinterval_size=5, min_subinterval_size=2) -> None:
        old_val = ind.evaluate()
        if ind.is_two_optimal_:
            return
        flipped_smthing = False
        for a in range(-max_subinterval_size, self.no_cities-max_subinterval_size):
            old_cost = sum(self.dist_matrix[ind.path_[i]][ind.path_[i+1]]
                           for i in range(a, a+min_subinterval_size-1, 1))
            for b in range(a+min_subinterval_size, min(self.no_cities, a+max_subinterval_size+1)):
                # old_cost = sum(self.dist_matrix[ind.path_[i]][ind.path_[i+1]]
                #                for i in range(a, b, 1))
                old_cost += self.dist_matrix[ind.path_[b-1]][ind.path_[b]]
                new_cost = self.dist_matrix[ind.path_[a]][ind.path_[b-1]] \
                    + sum(self.dist_matrix[ind.path_[i]][ind.path_[i-1]] for i in range(b-1, a+1, -1)) \
                    + self.dist_matrix[ind.path_[a+1]][ind.path_[b]]
                if new_cost < old_cost:
                    if a+1 < 0 and b >= 0:
                        temp = np.flip(np.concatenate(
                            (ind.path_[a+1:], ind.path_[:b])))
                        ind.path_[a+1:] = temp[:-(a+1)]
                        ind.path_[:b] = temp[-(a+1):]
                    else:
                        ind.path_[a+1:b] = np.flip(ind.path_[a+1:b])
                    flipped_smthing = True
                    ind.reset()
                    # if not ind.is_valid():
                    #     exit(1)
                    # if old_val < ind.evaluate():
                    #     exit(1)
                    break
        if not flipped_smthing:
            ind.is_two_optimal_ = True

# -------------------------------------------------------------------------------------
# Elimination operators

    # mu+lambda elimination
    def elimination_mu_plus_lambda(self) -> None:
        combined = self.population + self.offspring
        combined.sort(key=lambda ind: ind.evaluate())
        self.population = combined[:self.pop_size]

    # mu+lambda elimination with k-tournament crowding
    def elimination_mu_plus_lambda_k_crowding(self) -> None:
        combined = self.population + self.offspring
        combined.sort(key=lambda ind: ind.evaluate())

        for i in range(self.crowding_pop_count):
            combatants = random.sample(
                range(i+1, len(combined)), k=self.crowding_k) + [i+1]
            combined.pop(min(combatants,
                             key=lambda ind: combined[i].distance(combined[ind])))
        self.population = combined[:self.pop_size]

    # round-robin elimination
    def elimination_round_robin(self) -> None:
        combined = self.population + self.offspring
        best_guy = min(combined, key=lambda ind: ind.evaluate())
        # combined.sort(key = lambda ind: sum(int(ind.evaluate() < combatant.evaluate()) for combatant in random.choices(combined, k=self.round_robin_q_elimination)))
        scores = np.array([1+sum(int(ind.evaluate() < combatant.evaluate()) for combatant in random.choices(combined, k=self.round_robin_q_elimination))
                           for ind in combined])
        indicies = np.argsort(-scores)
        self.population = [combined[indicies[i]]
                           for i in range(self.no_cities)]
        if best_guy not in self.population:
            self.population[-1] = best_guy

    # round-robin elimination with k-crowding
    def elimination_round_robin_k_crowding(self) -> None:
        combined = self.population + self.offspring
        best_guy = min(combined, key=lambda ind: ind.evaluate())
        # combined.sort(key = lambda ind: sum(int(ind.evaluate() < combatant.evaluate()) for combatant in random.choices(combined, k=self.round_robin_q_elimination)))
        scores = np.array([1+sum(int(ind.evaluate() < combatant.evaluate()) for combatant in random.choices(combined, k=self.round_robin_q_elimination))
                           for ind in combined])
        indicies = np.argsort(-scores)
        combined = [combined[indicies[i]] for i in range(len(combined))]

        for i in range(self.crowding_pop_count):
            combatants = random.sample(
                range(i+1, len(combined)), k=self.crowding_k) + [i+1]
            combined.pop(min(combatants,
                             key=lambda ind: combined[i].distance(combined[ind])))
        self.population = combined[:self.pop_size]
        if best_guy not in self.population:
            self.population.insert(0, best_guy)
            del self.population[-1]


# -------------------------------------------------------------------------------------
# Individual class

    class Individual:
        path_: np.ndarray
        distance_matrix_: np.ndarray
        mut_prob_flip_: float
        mut_prob_swap_: float
        mut_prob_shuf_: float
        path_cost_: float
        edges: set
        no_cities_: int
        is_two_optimal_: bool

        def __init__(self, path, dist_matrix, mut_prob_flip, mut_prob_shuf):
            self.path_ = path
            self.distance_matrix_ = dist_matrix
            self.mut_prob_flip_ = mut_prob_flip
            self.mut_prob_shuf_ = mut_prob_shuf
            self.path_cost_ = None
            self.edges_ = None
            self.no_cities_ = self.path_.shape[0]
            self.is_two_optimal_ = False

        def flip_keep_better(self) -> bool:
            a, b = gamma_slice(self.no_cities_, 0.6)
            old_val = sum(self.distance_matrix_[self.path_[i]][self.path_[i+1]]
                          for i in range(a, b, 1))
            new_val = self.distance_matrix_[self.path_[a]][self.path_[b-1]] \
                + sum(self.distance_matrix_[self.path_[i]][self.path_[i-1]] for i in range(b-1, a+1, -1)) \
                + self.distance_matrix_[self.path_[a+1]][self.path_[b]]
            if old_val > new_val:
                self.path_[a+1:b] = np.flip(self.path_[a+1:b])
                self.reset()

        def try_to_mutate(self) -> bool:
            mutated = False
            if self.mut_prob_flip_ > np.random.uniform():
                self.mutate_flip()
                mutated = True
            if self.mut_prob_shuf_ > np.random.uniform():
                self.mutate_shuffle()
                mutated = True
            return mutated

        def mutate_flip(self) -> None:
            a, b = gamma_slice(self.no_cities_, 0.25)
            self.path_[a:b] = np.flip(self.path_[a:b])
            self.reset()

        def mutate_shuffle(self) -> None:
            a, b = gamma_slice(self.no_cities_, 0.05)
            np.random.shuffle(self.path_[a:b])
            self.reset()

        def reset(self) -> None:
            self.path_cost_ = None
            self.edges_ = None
            self.is_two_optimal_ = False

        def evaluate(self) -> float:
            if self.path_cost_ == None:
                self.path_cost_ = sum(self.distance_matrix_[self.path_[i-1]][self.path_[i]]
                                      for i in range(0, self.no_cities_))
            return self.path_cost_

        # evaluation from group phase
        def old_eval(self) -> float:
            if self.path_cost_ == None:
                self.path_cost_ = 0.0
                for i in range(1, self.no_cities_):
                    self.path_cost_ += self.distance_matrix_[
                        self.path_[i-1]][self.path_[i]]
                self.path_cost_ += self.distance_matrix_[
                    self.path_[-1]][self.path_[0]]
            return self.path_cost_

        # check if path is valid (debugging purpose)
        def is_valid(self) -> bool:
            if (len(self.path_) != self.no_cities_):
                return False
            return len(set(self.path_)) == len(self.path_)

        # return a set of tuples representing edges of the path
        def edges(self) -> set:
            if self.edges_ == None:
                self.edges_ = set((self.path_[i-1], self.path_[i])
                                  for i in range(0, self.no_cities_))
            return self.edges_

        def distance(self, that) -> int:
            return len(self.edges().difference(that.edges()))

    def validity_check(self) -> None:
        print("Population size: ", len(self.population))
        print("Offspring size: ", len(self.offspring))
        cities = len(self.population[-1].path_)
        print("No cities: ", cities)
        for ind in self.population:
            if (len(ind.path_) != cities):
                print("Individual with wrong path length! ", ind)
                exit(2)
            if (not ind.is_valid()):
                print("Invalid individual! ", ind)
                exit(1)

    def avg_distance(self) -> float:
        avg = sum(ind.distance(that)
                  for that in self.population for ind in self.population)
        avg /= self.pop_size**2 * self.no_cities
        print("Average realtive distance: ", int(avg*100), "%")
        return avg

    def avg_mutation_rate(self) -> float:
        avg = sum((ind.mut_prob_flip_ + ind.mut_prob_shuf_) / 2.0
                  for ind in self.population)
        avg /= self.pop_size
        print("Average mutation rate: ", int(avg*100), "%")
        return avg


# -------------------------------------------------------------------------------------
# Utility funcitons


def gamma_slice(arr_len, center_prc) -> int:
    slice_len = min(arr_len-1,
                    max(2, round(np.random.gamma(center_prc*arr_len))))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len


def normal_slice(arr_len, center_prc, spread_prc) -> int:
    slice_len = min(
        arr_len-1, max(2, round(np.random.normal(center_prc, spread_prc)*arr_len)))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len


def uniform_slice(arr_len) -> int:
    slice_len = min(
        arr_len-1, max(2, round(np.random.uniform()*arr_len)))
    a = random.randrange(0, arr_len-slice_len)
    return a, a + slice_len


# -------------------------------------------------------------------------------------
# main
if __name__ == "__main__":
    c = r0829194()
    c.optimize("tour929.csv")

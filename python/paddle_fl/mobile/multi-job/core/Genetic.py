import copy
import random
from itertools import accumulate
from bisect import bisect_right
import numpy as np

population_size = 10
cross_rate = 0.6
mutation_rate = 0.2
DNA_length = 30
generation = 1000

# Beta
O = [2 * (10 ** 5), 10 ** 5, 2 * (10 ** 4)]
O = [10 ** 6, 2 * (10 ** 5), 10 ** 4]


def initPopulation(device, greedy_scheme):
    population = []
    # Call greedy algorithm and add it solution to population
    scheme = greedy_scheme
    population.append(scheme)
    # Generate N-1 candidates randomly and add them to population
    for i in range(population_size - 1):
        life = random.sample(device, DNA_length)
        population.append(life)
    return population


def Variance(pre_clients, scheme, t):
    """
    :param pre_clients: 3 x 100
    :param scheme: 1 x 30
    :param t: O
    :return: 1 x 3
    """
    if scheme is not None:
        for d in scheme:
            pre_clients[d] += 1

    num_class = pre_clients
    S = sum(num_class)
    if S == 0:
        return 0
    else:
        num_class = np.array(num_class) / S  # 放大数据均方差的影响
        N = len(num_class)
        miu = sum(num_class) / N
        return sum((np.array(num_class) - miu) ** 2) / N * t


def get_reward(scheme, Pre_clients, client_time, num_job):
    """
    :param scheme: 1 x 10
    :param Pre_clients: 3 x 100
    :param num_job: 3
    :return: int
    """
    Pre = copy.deepcopy(Pre_clients[num_job])
    R = 0
    time = []
    for i in scheme:
        Pre[i] += 1
        time.append(client_time[i][num_job])
    V = Variance(Pre, None, O[num_job])
    T = max(time)
    R -= V + T
    return R


# Roulette Wheel Selection
def select(value, population):
    # Normalize fitness value
    value_sum = sum(value)
    fitness = [i / value_sum for i in value]
    # Create roulette wheel
    sum_fit = sum(fitness)
    wheel = list(accumulate([i / sum_fit for i in fitness]))
    # Select parent
    father_idx = bisect_right(wheel, random.random())
    father = population[father_idx]
    mother_idx = bisect_right(wheel, random.random())
    mother = population[mother_idx]
    return father, mother


# Partial-Mapped crossover
def PMX_cross(father, mother):
    father_copy = copy.deepcopy(father)
    mother_copy = copy.deepcopy(mother)
    # Cross location, (idex1 < idex2)
    index1 = random.randint(0, DNA_length - 1)
    index2 = random.randint(index1, DNA_length - 1)
    # Reccord cross fragment
    fragment1 = father_copy[index1:index2]
    fragment2 = mother_copy[index1:index2]
    # Cross
    father_copy[index1:index2] = mother_copy[index1:index2]
    mother_copy[index1:index2] = father_copy[index1:index2]
    # Record validate head and tail
    child1_head = []
    child1_tail = []
    child2_head = []
    child2_tail = []
    # Validate child1 head
    for i in father_copy[:index1]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        child1_head.append(i)
    # Validate child1 tail
    for i in father_copy[index2:]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        child1_tail.append(i)
    # Validate child2 head
    for i in mother_copy[:index1]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        child2_head.append(i)
    # Validate child2 tail
    for i in mother_copy[index2:]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        child2_tail.append(i)

    child1 = child1_head + fragment2 + child1_tail
    child2 = child2_head + fragment1 + child2_tail
    return child1, child2


# Three point mutation
def mutation(child, device):
    for i in range(0, 30, 10):
        candidate_list = []
        for d in device:
            if d not in child:
                candidate_list.append(d)
        index = random.randint(i, i + 9)
        child[index] = random.choice(candidate_list)
    return child


# Gene algorithm
def Gene(device, Pre_clients, client_time, greedy_scheme):
    population = initPopulation(device, greedy_scheme)
    for G in range(generation):
        # Calculate fitness values for candidates in population
        fitness = []  # get per individual reward
        for i in population:
            R_life = 0
            for s in range(0, 30, 10):
                scheme = i[s:s + 9]
                R = get_reward(scheme, Pre_clients, client_time, s // 10)
                R_life += R
            fitness.append(R_life)
        # Perform reproduction
        if random.random() < cross_rate:
            # Select candidates using selection operator
            father, mother = select(fitness, population)
            # Create new solution using crossover operator
            child1, child2 = PMX_cross(father, mother)
            # Create new solution using mutation operator
            if random.random() < mutation_rate:
                child1 = mutation(child1, device)
                child2 = mutation(child2, device)
            population.append(child1)
            population.append(child2)
            # Replacement
            for i in population[population_size:population_size + 2]:
                R_life = 0
                for s in range(0, 30, 10):
                    scheme = i[s:s + 9]
                    R = get_reward(scheme, Pre_clients, client_time, s // 10)
                    R_life += R
                fitness.append(R_life)
            for count in range(2):
                worst_fit = min(fitness)
                worst_fit_idx = fitness.index(worst_fit)
                fitness.remove(worst_fit)
                population.pop(worst_fit_idx)
            # if Termination?

        # Best solution
        best_life_fit = max(fitness)
        best_life_idx = fitness.index(best_life_fit)
    return population[best_life_idx]
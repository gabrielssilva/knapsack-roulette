'''
Genetic Algorithm implementation for the Multi-dimensional Knapsack Problem
using Roulette Wheel Selection.
'''
import random
import numpy as np


def _generate_initial_population(population_size, num_of_items):
    '''
    Generate the initial population of solutions.

    Args:
        population_size (int): Size of the population to generate
        num_of_items (int): Number of items in the knapsack problem

    Returns:
        numpy.ndarray: Array of shape (population_size, num_of_items) containing binary values
                      (1 if item is selected, 0 if not)
    '''
    solution_matrix = np.random.randint(0, 2, (population_size, num_of_items))
    return solution_matrix


def _f(solution, benefits, items, dimensions, excess_factor=2):
    '''
    Evaluate the fitness of a solution.

    Args:
        solution (numpy.ndarray): Binary array representing a solution
        benefits (numpy.ndarray): Array of benefit values for each item
        items (numpy.ndarray): Array of shape (num_dimensions, num_items) containing weights
        dimensions (numpy.ndarray): Array of capacity constraints for each dimension
        excess_factor (float): Factor to penalize solutions that exceed constraints

    Returns:
        float: Fitness value of the solution (higher is better)
    '''
    total_profit = np.dot(solution, benefits)

    for i in range(len(dimensions)):
        # For every dimension, check if the solution fits
        total_size = np.dot(solution, items[i])

        if total_size > dimensions[i]:
            # Solution does not fit, we need to adjust
            excess = total_size * excess_factor
            penalty = total_profit * ((excess - dimensions[i]) / dimensions[i])

            return (total_profit - penalty)

    return total_profit


def _select_parents(population, fitness_values):
    '''
    Select parents for the next generation using roulette wheel selection.

    Args:
        population (numpy.ndarray): Array of shape (population_size, num_items) containing solutions
        fitness_values (numpy.ndarray): Array of fitness values for each solution

    Returns:
        tuple: Two selected solutions (numpy.ndarray) to be used as parents
    '''
    # Scale fitness values to avoid non-positive values
    min_fitness = np.min(fitness_values)
    scaled_fitness = fitness_values + abs(min_fitness) + 1e-10

    probabilities = scaled_fitness / scaled_fitness.sum()
    first_parent_index = np.random.choice(len(population), p=probabilities)
    second_parent_index = np.random.choice(len(population), p=probabilities)

    while first_parent_index == second_parent_index:
        second_parent_index = np.random.choice(len(population), p=probabilities)

    return (population[first_parent_index], population[second_parent_index])


def _crossover(parent1, parent2):
    '''
    Perform single-point crossover between two parents.

    Args:
        parent1 (numpy.ndarray): First parent solution
        parent2 (numpy.ndarray): Second parent solution

    Returns:
        tuple: Two children created through crossover
    '''
    num_of_items = len(parent1)
    crossover_point = np.random.randint(1, num_of_items)

    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2


def _mutate(child, mutation_rate):
    '''
    Apply mutation to a solution by flipping bits with given probability.

    Args:
        child (numpy.ndarray): Solution to mutate
        mutation_rate (float): Probability of flipping each bit

    Returns:
        numpy.ndarray: Mutated solution
    '''
    mutation_mask = np.random.random(len(child)) < mutation_rate
    child[mutation_mask] = 1 - child[mutation_mask] # Use boolean indexing to flip bits
    return child


def _next_generation(population, fitness_values, crossover_rate=0.5, mutation_rate=0.1, elite_ratio=0.1):
    '''
    Generate the next generation through parent selection and crossover.

    Args:
        population (numpy.ndarray): Array of shape (population_size, num_items) containing solutions
        fitness_values (numpy.ndarray): Array of fitness values for each solution
        crossover_rate (float): Probability of performing crossover between parents (default: 0.5)
        mutation_rate (float): Probability of flipping each bit in a solution (default: 0.1)
        elite_ratio (float): Proportion of best solutions to preserve (default: 0.1, i.e., 10%)

    Returns:
        numpy.ndarray: Array of shape (population_size, num_items) containing the new generation
    '''
    population_size, num_of_items = population.shape
    elite_size = max(1, int(population_size * elite_ratio))
    num_of_couples = (population_size - elite_size) // 2

    next_generation = np.empty((population_size, num_of_items), dtype=int)

    # Save the best solutions through elitism
    elite_indexes = np.argsort(fitness_values)[-elite_size:]
    next_generation[:elite_size] = population[elite_indexes]

    # Generate remaining solutions through crossover and mutation
    for i in range(num_of_couples):
        parent1, parent2 = _select_parents(population, fitness_values)

        if np.random.random() < crossover_rate:
            child1, child2 = _crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        next_generation[elite_size + 2*i] = _mutate(child1, mutation_rate)
        next_generation[elite_size + 2*i + 1] = _mutate(child2, mutation_rate)

    return next_generation


def solve_knapsack(
    benefits,
    items,
    constraints,
    population_size=100,
    num_of_generations=10_000,
    crossover_rate=0.8,
    mutation_rate=0.05,
    elite_ratio=0.02
):
    '''
    Solve the multi-dimensional knapsack problem using a genetic algorithm.

    Args:
        benefits (numpy.ndarray): Array of benefit values for each item
        items (numpy.ndarray): Array of shape (num_dimensions, num_items) containing weights
        constraints (numpy.ndarray): Array of capacity constraints for each dimension
        population_size (int): Size of the population to maintain (will be adjusted to even if odd)
        num_of_generations (int): Number of generations to evolve
        crossover_rate (float): Probability of performing crossover between parents
        mutation_rate (float): Probability of flipping each bit in a solution
        elite_ratio (float): Proportion of best solutions to preserve

    Returns:
        tuple: Two selected solutions (numpy.ndarray) to be used as parents
    '''
    # Round population size up to next even number if it's odd
    if population_size % 2 != 0:
        population_size += 1

    num_of_items = len(benefits)
    population = _generate_initial_population(population_size, num_of_items)
    fitness_values = np.array([_f(solution, benefits, items, constraints) for solution in population])

    result = {
        'best_fitness': 0,
        'best_solution': None,
        'generations_plot': [],
    }

    for i in range(num_of_generations):
        population = _next_generation(population, fitness_values, crossover_rate, mutation_rate, elite_ratio)
        fitness_values = np.array([_f(solution, benefits, items, constraints) for solution in population])
        best_fitness = np.max(fitness_values)

        progress = ((i+1) / num_of_generations) * 100
        print(f'\rGeneration {i+1}/{num_of_generations} ({progress:.2f}%): {best_fitness}', end='', flush=True)

        result['generations_plot'].append((i, best_fitness))

        if best_fitness > result['best_fitness']:
            best_solution_index = np.argmax(fitness_values)
            result['best_fitness'] = best_fitness
            result['best_solution'] = population[best_solution_index]

    print('')
    return result
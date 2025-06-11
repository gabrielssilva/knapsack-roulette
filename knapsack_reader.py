import os
import numpy as np
from itertools import islice

def _get_knapsack_stream(file_path):
    '''
    Read a knapsack problem file and yield each space-separated value as an integer.

    Args:
        file_path (str): Path to the knapsack problem file

    Yields:
        int: Each space-separated value from the file, converted to integer
    '''
    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and split by spaces
            values = line.strip().split()
            for value in values:
                yield int(value)


def _read_problem_values(knapsack_values):
    '''
    Parse a single multi-dimensional knapsack problem.

    The problem format is:
    1. First three values: number of items (n), number of dimensions (d), and an unused value
    2. Next n values: benefits for each item
    3. Next n*d values: weights for each item in each dimension (d rows of n values)
    4. Next d values: capacity constraints for each dimension

    Args:
        knapsack_values: A generator yielding integer values from the problem file

    Returns:
        tuple: A tuple containing:
            - benefits (numpy.ndarray): Array of n benefit values, one for each item
            - items (list): List of d numpy.ndarray, where each array contains n weights
                           (items[i][j] is the weight of item j in dimension i)
            - constraints (numpy.ndarray): Array of d capacity constraints, one for each dimension
    '''
    num_of_items, num_of_dimensions, _ = islice(knapsack_values, 3)
    benefits = list(islice(knapsack_values, num_of_items))
    items = []

    for _ in range(num_of_dimensions):
        item_list = list(islice(knapsack_values, num_of_items))
        items.append(np.array(item_list))

    dimensions = list(islice(knapsack_values, num_of_dimensions))

    return (np.array(benefits), items, np.array(dimensions))


def generate_knapsack_problems(file_path):
    '''
    Generate all knapsack problems from a file.

    Each problem in the file follows this format:
    1. First three values: number of items (n), number of dimensions (d), and an unused value
    2. Next n values: benefits for each item
    3. Next n*d values: weights for each item in each dimension
    4. Next d values: capacity constraints for each dimension

    Args:
        file_path (str): Path to the file containing knapsack problems

    Yields:
        tuple: A tuple containing:
            - benefits (numpy.ndarray): Array of n benefit values, one for each item
            - items (list): List of d numpy.ndarray, where each array contains n weights
                           (items[i][j] is the weight of item j in dimension i)
            - constraints (numpy.ndarray): Array of d capacity constraints, one for each dimension
    '''
    knapsack_values = _get_knapsack_stream(file_path)
    amount_of_problems = int(next(knapsack_values))

    for _ in range(amount_of_problems):
        (benefits, items, constraints) = _read_problem_values(knapsack_values)
        yield (benefits, items, constraints)
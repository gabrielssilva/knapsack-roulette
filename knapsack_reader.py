import os
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


def _retrieve_all_items(num_of_items, dimensions, knapsack_values):
    '''
    Parse all items for a multi-dimensional knapsack problem.

    The problem format for items is:
    1. n values: benefits for each item
    2. n*d values: weights for each item in each dimension (d rows of n values)

    Args:
        num_of_items (int): Number of items in the problem
        dimensions (int): Number of dimensions (constraints) in the problem
        knapsack_values: A generator yielding integer values from the problem file

    Returns:
        list: A list of n tuples, each containing:
            - benefit (int): The benefit value for the item
            - weights (tuple): A tuple of ordered weights, one for each dimension
    '''
    benefits = list(islice(knapsack_values, num_of_items))
    weights = []

    for _ in range(dimensions):
        weights.append(list(islice(knapsack_values, num_of_items)))

    grouped_weights = zip(*weights)
    return list(zip(benefits, grouped_weights))


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
            - items: List of (benefit, weights) tuples for each item
            - constraints: List of capacity constraints for each dimension
    '''
    knapsack_values = _get_knapsack_stream(file_path)
    amount_of_problems = int(next(knapsack_values))

    for _ in range(amount_of_problems):
        num_of_items, dimensions, _ = islice(knapsack_values, 3)
        items = _retrieve_all_items(num_of_items, dimensions, knapsack_values)
        constraints = list(islice(knapsack_values, dimensions))
        yield (items, constraints)
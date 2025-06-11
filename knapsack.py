from knapsack_reader import generate_knapsack_problems
import argparse
import roulette_ga
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Solve knapsack problems using Genetic Algorithm')
    parser.add_argument('--file', '-f', required=True, help='Path to the knapsack problem file')
    return parser.parse_args()


def is_solution_valid(solution, items, dimensions):
    solution_dimensions = np.zeros(len(dimensions))

    for i in range(len(dimensions)):
        solution_dimensions[i] = np.dot(solution, items[i])

    return np.all(solution_dimensions <= dimensions)


def main():
    args = parse_args()
    first_problem = next(generate_knapsack_problems(args.file))
    benefits, items, dimensions = first_problem

    result = roulette_ga.solve_knapsack(benefits, items, dimensions)
    print(f'Best solution: {result['best_solution']}')
    print(f'valid?: {is_solution_valid(result['best_solution'], items, dimensions)}')


if __name__ == '__main__':
    main()

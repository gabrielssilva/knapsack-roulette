from knapsack_reader import generate_knapsack_problems
import argparse
import roulette_ga
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Solve knapsack problems using Genetic Algorithm')
    parser.add_argument('--file', '-f', required=True, help='Path to the knapsack problem file')
    parser.add_argument('--population-size', '-p', type=int, default=100, help='Size of the population')
    parser.add_argument('--generations', '-g', type=int, default=10_000, help='Number of generations to evolve')
    parser.add_argument('--crossover-rate', '-c', type=float, default=0.8, help='Probability of crossover')
    parser.add_argument('--mutation-rate', '-m', type=float, default=0.05, help='Probability of mutation')
    parser.add_argument('--elite-ratio', '-e', type=float, default=0.05, help='Proportion of best solutions to preserve')
    parser.add_argument('--excess-factor', '-x', type=float, default=5, help='Factor to penalize solutions that exceed constraints')
    parser.add_argument('--num_of_problems', '-n', type=int, default=1, help='Number of problems to solve from the input file')
    return parser.parse_args()


def _is_solution_valid(solution, items, dimensions):
    """Check if a solution is valid by verifying if it respects all dimension constraints.

    Args:
        solution (numpy.ndarray): Binary array representing the solution (1 = item included, 0 = item excluded)
        items (numpy.ndarray): Array of items with their weights for each dimension
        dimensions (numpy.ndarray): Array of capacity constraints for each dimension

    Returns:
        bool: True if the solution is valid (respects all constraints), False otherwise
    """
    solution_dimensions = np.zeros(len(dimensions))

    for i in range(len(dimensions)):
        solution_dimensions[i] = np.dot(solution, items[i])

    return np.all(solution_dimensions <= dimensions)


def _plot_generations(index, generations_plot):
    os.makedirs('charts', exist_ok=True)

    plt.plot([x[0] for x in generations_plot], [x[1] for x in generations_plot])
    plt.xlabel('Generation')
    plt.ylabel('Best fitness')
    plt.savefig(f'charts/generations_{index + 1}.png')
    plt.close()


def _plot_problems_fitness(problem_fitnesses):
    x = range(1, len(problem_fitnesses) + 1)  # Start from 1 for problem numbers

    plt.bar(x, problem_fitnesses)
    plt.xlabel('Problem Number')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness by Problem')
    plt.xticks(x)  # Force integer ticks
    plt.savefig('charts/problems_fitness.png')
    plt.close()


def main():
    args = parse_args()
    problems = generate_knapsack_problems(args.file)
    problem_fitnesses = []

    try:
        for i in range(args.num_of_problems):
            benefits, items, dimensions = next(problems)
            print('\n') if i > 0 else None
            print(f'Solving problem {i+1}/{args.num_of_problems}')

            max_attempts = 5
            attempt = 1
            valid_solution = False

            while not valid_solution and attempt <= max_attempts:
                if attempt > 1:
                    print(f'Solution was invalid, let\'s try again. Attempt {attempt}/{max_attempts}')

                result = roulette_ga.solve_knapsack(
                    benefits,
                    items,
                    dimensions,
                    population_size=args.population_size,
                    num_of_generations=args.generations,
                    crossover_rate=args.crossover_rate,
                    mutation_rate=args.mutation_rate,
                    elite_ratio=args.elite_ratio,
                    excess_factor=args.excess_factor
                )

                valid_solution = _is_solution_valid(result['best_solution'], items, dimensions)
                attempt += 1

            if valid_solution:
                print('Best solution:')
                print(result['best_solution'])
                print(f'Best fitness: {result['best_fitness']}')

                _plot_generations(i, result['generations_plot'])
                problem_fitnesses.append(result['best_fitness'])
            else:
                print(f'Failed to find a valid solution after {max_attempts} attempts')
                problem_fitnesses.append(0)

    except StopIteration:
        print(f'Reached end of file after solving {i + 1} problems')
    except KeyboardInterrupt:
        print('\n\nOkay, I\'ll stop here.')
        print(f'Successfully solved {i} problems, saved details to charts/')
    finally:
        if problem_fitnesses:
            _plot_problems_fitness(problem_fitnesses)


if __name__ == '__main__':
    main()

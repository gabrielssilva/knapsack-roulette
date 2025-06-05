from knapsack_reader import generate_knapsack_problems
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Solve knapsack problems using Genetic Algorithm')
    parser.add_argument('--file', '-f', required=True, help='Path to the knapsack problem file')
    return parser.parse_args()


def main():
    args = parse_args()
    first_problem = next(generate_knapsack_problems(args.file))
    items, constraints = first_problem
    print(f'Items: {items}')
    print(f'Constraints: {constraints}')


if __name__ == '__main__':
    main()

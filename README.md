# Multi-Dimensional Knapsack Problem Solver

A genetic algorithm implementation for solving the multi-dimensional knapsack problem (MDKP). This tool uses a roulette wheel selection genetic algorithm to find optimal solutions for knapsack problems with multiple constraints.

## Overview

The solver implements a genetic algorithm for the multi-dimensional knapsack problem, where each item has:
- A benefit value
- Multiple weight values (one for each dimension)
- Multiple capacity constraints (one for each dimension)

The algorithm features:
- Roulette wheel selection for parent selection
- Single-point crossover
- Bit-flip mutation
- Elitism to preserve best solutions
- Penalty-based fitness function for constraint handling
- Support for multiple problems in a single input file

## Setup

### Requirements
- Python 3.x
- numpy
- matplotlib

### Installation Steps

1. Optional: Create a virtual environment:
```bash
python -m venv venv
```

2. Case you created a virutal environemnt, activate it:
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   - On Unix or MacOS:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The tool can be run from the command line with various parameters. A sample problem file `files/mknapcb1.txt` is provided in the repository.

```bash
python knapsack.py -f files/mknapcb1.txt [options]
```

### Required Arguments
- `-f, --file`: Path to the knapsack problem file

### Optional Arguments
- `-p, --population-size`: Size of the population (default: 100)
- `-g, --generations`: Number of generations to evolve (default: 10000)
- `-c, --crossover-rate`: Probability of crossover (default: 0.8)
- `-m, --mutation-rate`: Probability of mutation (default: 0.05)
- `-e, --elite-ratio`: Proportion of best solutions to preserve (default: 0.05)
- `-x, --excess-factor`: Factor to penalize solutions that exceed constraints (default: 5)
- `-n, --num_of_problems`: Number of problems to solve from the input file (default: 1)

### Example
```bash
# Run with default parameters
python knapsack.py -f files/mknapcb1.txt

# Run with custom parameters
python knapsack.py -f files/mknapcb1.txt -p 200 -g 5000 -c 0.7 -m 0.05 -e 0.02 -x 2.5 -n 3
```

## Input File Format

The input file should follow this format:
1. First line: Number of problems in the file
2. For each problem:
   - First three values: number of items (n), number of dimensions (d), and an unused value
   - Next n values: benefits for each item
   - Next n*d values: weights for each item in each dimension
   - Next d values: capacity constraints for each dimension

## Output

### Console Output
Example output:
```
Solving problem 1/3
Generation 10000/10000 (100.00%): 42.0
Best solution:
[1, 0, 1, 0, 1]
Best fitness: 42.0

Solving problem 2/3
...
```

### Charts
The tool generates plots showing the evolution of the best fitness value over generations. These are saved in a `charts` directory and named following the `charts/generations_{n}.png` convention. It'll also generate a bar plot displaying the best found fitness for every result (`charts/problems_fitness.png`).

## References

The sample problem file `mknapcb1.txt` is from the OR-Library, a collection of test data sets for Operations Research problems. More information about the multi-dimensional knapsack problem test instances can be found at [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html).
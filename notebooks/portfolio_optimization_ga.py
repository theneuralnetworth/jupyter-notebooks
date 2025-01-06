# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# %%
class PortfolioOptimizationGA:
    def __init__(self, returns, sectors, covariance_matrix, n_assets, risk_aversion, min_return, max_weight, min_weight, excluded_sectors,
                 weight_penalty=1000, return_penalty=1000, sector_penalty=1000, optimal_fitness_result=1000, no_improvement_generations=50,
                 generations=1000, crossover_rate=0.7, mutation_rate=0.1):
        # Existing parameters
        self.returns = returns
        self.sectors = sectors
        self.cov_matrix = covariance_matrix
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.min_return = min_return
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.excluded_sectors = excluded_sectors
        
        # Penalty values for each constraint
        self.weight_penalty = weight_penalty
        self.return_penalty = return_penalty
        self.sector_penalty = sector_penalty
        
        # New stopping conditions parameters
        self.optimal_fitness_result = optimal_fitness_result  # The fitness threshold to stop
        self.no_improvement_generations = no_improvement_generations  # No improvement generations threshold
        
        # Algorithm parameters
        self.population_size = 100
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = 5

    def fitness(self, portfolio):
        # Calculate expected return of portfolio
        R = np.dot(self.returns, portfolio)
        
        # Calculate portfolio variance (risk)
        portfolio_variance = np.dot(portfolio.T, np.dot(self.cov_matrix, portfolio))
        sigma_squared = portfolio_variance
        
        # Calculate fitness (expected return - λ * risk)
        fitness_value = R - self.risk_aversion * sigma_squared
        
        # Penalties for constraints
        # Penalty for weight constraints
        if any(portfolio < self.min_weight) or any(portfolio > self.max_weight):
            fitness_value -= self.weight_penalty  # Apply weight penalty

        # Penalty for return constraints
        if R < self.min_return:
            fitness_value -= self.return_penalty  # Apply return penalty

        # Penalty for sector constraints
        
        for i, sector in enumerate(self.sectors):
            if sector in excluded_sectors and portfolio[i] > 0:
                fitness_value -= self.sector_penalty

        return fitness_value

    def selection(self, population):
        # Tournament selection
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament = population[tournament_indices]
        tournament_fitness = [self.fitness(individual) for individual in tournament]
        winner = tournament[np.argmax(tournament_fitness)]
        return winner

    def crossover(self, parent1, parent2):
        # One-point crossover with crossover rate
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.n_assets)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            return child
        else:
            # If no crossover occurs, return one of the parents unchanged
            return parent1

    def mutate(self, individual):
        # Random mutation with mutation rate
        if np.random.rand() < self.mutation_rate:
            mutate_index = np.random.randint(self.n_assets)
            mutation_value = np.random.uniform(-0.2, 0.2)
            individual[mutate_index] += mutation_value

            if(individual[mutate_index]<=0):
                individual[mutate_index] = 0
                
            # Ensure the portfolio is still valid
            individual = self.normalize_portfolio(individual)
        return individual

    def normalize_portfolio(self, portfolio):
        # Set negative weights to zero
        portfolio = np.maximum(portfolio, 0)
        
        # Normalize the portfolio to sum to 1
        total_weight = np.sum(portfolio)
        if total_weight > 0:
            portfolio = portfolio / total_weight
        else:
            # If all weights are zero, reinitialize to uniform distribution
            portfolio = np.ones(self.n_assets) / self.n_assets
        return portfolio

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.rand(self.n_assets)
            individual = self.normalize_portfolio(individual)
            population.append(individual)
        return np.array(population)

    def run(self):
        # Initialize population
        population = self.generate_initial_population()

        best_fitness = -np.inf  # Initialize the best fitness to a very low value
        no_improvement_count = 0  # Count generations without fitness improvement

        for gen in range(self.generations):
            new_population = []

            # Create new population
            for _ in range(self.population_size // 2):
                parent1 = self.selection(population)
                parent2 = self.selection(population)

                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            population = np.array(new_population)

            # Check fitness of the best individual in the population
            current_best_fitness = max([self.fitness(individual) for individual in population])
            print(f"Generation {gen}: Best fitness = {current_best_fitness}")
            # Check if we've reached the optimal fitness
            if current_best_fitness >= self.optimal_fitness_result:
                print(f"Optimal fitness result reached: {current_best_fitness}")
                break

            # Track if there was an improvement in fitness
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_count = 0  # Reset counter if fitness improved
            else:
                no_improvement_count += 1

            # If no improvement after a set number of generations, stop the algorithm
            if no_improvement_count >= self.no_improvement_generations:
                print(f"No improvement in fitness after {self.no_improvement_generations} generations.")
                break

        # Return the best portfolio found
        best_portfolio = max(population, key=self.fitness)
        return best_portfolio

# %%
returns = np.array([0.08, 0.12, 0.10, 0.07, 0.15, 0.09, 0.06, 0.11, 0.14, 0.05]) 
sectors = [
    "Technology", "Technology", "Financial", "Energy", 
    "Consumer", "Industrial", "Real Estate", "Utilities", 
    "Communications", "Materials"
]

cov_matrix = np.array([
    [0.04, 0.02, 0.03, 0.01, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02],
    [0.02, 0.05, 0.02, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01],
    [0.03, 0.02, 0.06, 0.02, 0.01, 0.02, 0.02, 0.03, 0.01, 0.01],
    [0.01, 0.02, 0.02, 0.03, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02],
    [0.02, 0.01, 0.01, 0.01, 0.04, 0.01, 0.03, 0.02, 0.01, 0.01],
    [0.01, 0.02, 0.02, 0.01, 0.01, 0.05, 0.02, 0.01, 0.02, 0.01],
    [0.02, 0.01, 0.02, 0.02, 0.03, 0.02, 0.04, 0.01, 0.02, 0.02],
    [0.01, 0.02, 0.03, 0.01, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02],
    [0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02, 0.05, 0.02],
    [0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03]
])

n_assets = 10
risk_aversion = 2
min_return = 0.05
max_weight = 0.3
min_weight = 0.05
excluded_sectors = ["Technology"]

weight_penalty = 100
return_penalty = 100
sector_penalty = 1000

optimal_fitness_result = 1000
no_improvement_generations = 2000

generations = 5000
crossover_rate = 0.7
mutation_rate = 0.2

ga = PortfolioOptimizationGA(returns, sectors, cov_matrix, n_assets, risk_aversion, min_return, max_weight, min_weight, excluded_sectors,
                             weight_penalty, return_penalty, sector_penalty, optimal_fitness_result, no_improvement_generations,
                             generations, crossover_rate, mutation_rate)
best_portfolio = ga.run()

print("Optimized Portfolio Weights:", best_portfolio)


# %%
app = FastAPI()

class PortfolioRequest(BaseModel):
    returns: list[float]
    sectors: list[str]
    covariance_matrix: list[list[float]]
    n_assets: int
    risk_aversion: float
    min_return: float
    max_weight: float
    min_weight: float
    excluded_sectors: list[str]
    weight_penalty: float = 1000
    return_penalty: float = 1000
    sector_penalty: float = 1000
    optimal_fitness_result: float = 1000
    no_improvement_generations: int = 50
    generations: int = 1000
    crossover_rate: float = 0.7
    mutation_rate: float = 0.1

@app.post("/optimize-portfolio/")
async def optimize_portfolio(request: PortfolioRequest):
    try:
        # Convert lists to numpy arrays
        returns = np.array(request.returns)
        covariance_matrix = np.array(request.covariance_matrix)

        # Initialize the genetic algorithm
        ga = PortfolioOptimizationGA(
            returns=returns,
            sectors=request.sectors,
            covariance_matrix=covariance_matrix,
            n_assets=request.n_assets,
            risk_aversion=request.risk_aversion,
            min_return=request.min_return,
            max_weight=request.max_weight,
            min_weight=request.min_weight,
            excluded_sectors=request.excluded_sectors,
            weight_penalty=request.weight_penalty,
            return_penalty=request.return_penalty,
            sector_penalty=request.sector_penalty,
            optimal_fitness_result=request.optimal_fitness_result,
            no_improvement_generations=request.no_improvement_generations,
            generations=request.generations,
            crossover_rate=request.crossover_rate,
            mutation_rate=request.mutation_rate,
        )

        # Run the optimization
        best_portfolio = ga.run()

        # Return the optimized portfolio
        return {"optimized_portfolio_weights": best_portfolio.tolist()}

    except Exception as e:
        return {"error": str(e)}

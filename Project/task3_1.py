import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *

# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500 # Number of simulation steps
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait # Controller to be used for simulation da para mudar

MUTATION_RATE = 0.2  # Probability of mutation
CROSSOVER_RATE = 0.5  # Probability of crossover
POPULATION_SIZE = 20  # Number of robot structures per generation

# ---- POPULATION GENERATION ----
def create_population():
    population = []
    for _ in range(POPULATION_SIZE):
        grid_size = (
            random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]),
            random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1])
        )
        random_robot, _ = sample_robot(grid_size)
        population.append(random_robot)
        
    #print(population)
    return population



# Formas de mudar a fitness fuction
# ✅ Maximizar a distância percorrida pelo robô em vez de apenas a soma das recompensas.
# ✅ Penalizar estruturas muito grandes para favorecer designs mais eficientes.
# ✅ Adicionar estabilidade para evitar robôs que tombam rapidamente.

def evaluate_fitness(robot_structure, view=False):    
    try:
        connectivity = get_full_connectivity(robot_structure)
  
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        t_reward = 0
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        for t in range(STEPS):  
            # Update actuation before stepping
            actuation = CONTROLLER(action_size,t)
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward
    except (ValueError, IndexError) as e:
        return 0.0
    
#mutação de ponto aleatória
def mutate(robot_structure):
    mutated = copy.deepcopy(robot_structure)
    if random.random() < MUTATION_RATE:
        x, y = random.randint(0, 4), random.randint(0, 4)
        mutated[x, y] = random.choice(VOXEL_TYPES)
    return mutated

#Crossover de um ponto aleatório
def crossover(robot_structure1, robot_structure2):
    if random.random() < CROSSOVER_RATE:
        x, y = random.randint(0, 4), random.randint(0, 4)
        child = copy.deepcopy(robot_structure1)
        child[x:, y:] = robot_structure2[x:, y:]
        return child
    return robot_structure1


# ---- EVOLUTIONARY ALGORITHM ----
def evolve():
    population = create_population()
    best_robot = None
    best_fitness = -float('inf')
    
    for generation in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_fitness(robot) for robot in population]
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort by fitness (descending)
        population = [population[i] for i in sorted_indices]
        
        # Elite selection
        if fitness_scores[sorted_indices[0]] > best_fitness:
            best_fitness = fitness_scores[sorted_indices[0]]
            best_robot = population[0]
        
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Best Fitness: {best_fitness}")
        
        new_population = population[:POPULATION_SIZE // 2]  # Keep the best half
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(population[:10], 2)  # Select from top 10
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    return best_robot, best_fitness


def main():
    best_robot, best_fitness = evolve()
    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    print(best_fitness)
    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
        i += 1
    utils.create_gif(best_robot, filename='random_search.gif', scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)

if __name__ == '__main__':
    #main()
    population = create_population()
    

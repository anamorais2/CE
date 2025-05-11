
import numpy as np
import pandas as pd
import random
import gymnasium as gym
from evogym.envs import *    
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import utils
import torch
import os
import csv
from datetime import datetime

NUM_GENERATIONS = 100
POP_SIZE = 50
SIGMA_INIT = 0.2
STEPS = 500
SCENARIOS = ['DownStepper-v0', 'ObstacleTraverser-v0']
SEEDS = [51,52,53,54,55]


robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connectivity = get_full_connectivity(robot_structure)

# --- Funções auxiliares de manipulação de pesos ---
def flatten_weights(weights_list):
    """Transforma uma lista de arrays de pesos em um vetor unidimensional"""
    return np.concatenate([w.flatten() for w in weights_list])

def structure_weights(flat_weights, model):
    """Transforma um vetor unidimensional em uma lista de arrays com as formas originais"""
    structured_weights = []
    current_idx = 0
    
    for param in model.parameters():
        shape = param.shape
        param_size = np.prod(shape)
        param_weights = flat_weights[current_idx:current_idx + param_size]
        structured_weights.append(param_weights.reshape(shape))
        current_idx += param_size
        
    return structured_weights

# --- Função para salvar os dados de cada geração ---
def save_generation_data(generation, population, fitness_scores, scenario, controller_name, seed, parameters):
    """
    Salva os dados de cada geração em um arquivo CSV
    """
    folder = f"results_seed_{seed}/{controller_name}_{scenario}_DE"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Cabeçalho com metadados da run (apenas na primeira geração)
        if generation == 0:
            writer.writerow(["# ALGORITHM", parameters.get("algorithm", "DE")])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        # Cabeçalho dos dados por indivíduo
        writer.writerow(["Index", "Fitness", "Reward", "Weights"])

        for i, (weights, fitness) in enumerate(zip(population, fitness_scores)):
            weights_str = str(weights)  # Converter pesos para string para armazenar no CSV
            writer.writerow([i, -fitness, fitness, weights_str])
            
# --- Função para salvar os resultados finais num Excel ---            
def save_results_to_excel(controller, best_fitness, scenario, population_size, num_generations, execution_time, seed, controller_weights, filename='task3_2_Results_Complete.xlsx'):
    """
    Salva os resultados em um arquivo Excel, incluindo os pesos e bias do controlador
    """
    # Converter os pesos e bias para uma string 
    weights_str = str(controller_weights)

    new_data = {
        'Scenario': [scenario],
        'Controller': [controller.__name__],
        'Population Size': [population_size],
        'Number of Generations': [num_generations],
        'Best Fitness': [best_fitness],
        'Execution Time (s)': [execution_time],
        'Seed': [seed],
        'Algorithm': ["DE"],
        'Controller Weights': [weights_str]
    }

    new_df = pd.DataFrame(new_data)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_excel(filename, index=False)
    print(f"Resultados salvos em {filename}")


# --- Função de Fitness ---
def evaluate_fitness(weights,scenario, brain, view=False):
        set_weights(brain, weights)  # Load weights into the network
        env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        sim = env
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]  # Get initial state
        t_reward = 0
        for t in range(STEPS):  
            # Update actuation before stepping
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = brain(state_tensor).detach().numpy().flatten() # Get action
            if view:
                viewer.render('screen') 
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward 

# ---- Funções de Inicialização e Evolução ----
def initialize_controller_population(input_size, output_size, size=POP_SIZE):
    base_model = NeuralController(input_size, output_size)
    flat_weights = flatten_weights(get_weights(base_model))
    dim = len(flat_weights)

    population = []
    for _ in range(size):
        # Inicializa em torno de valores pequenos
        individual = np.random.normal(0, SIGMA_INIT, dim)
        structured = structure_weights(individual, base_model)
        population.append(structured)

    return population, base_model

def evolve_controllers(population, scenario, fitnesses, base_model, F=0.5, CR=0.7):
    new_population = []
    dim = len(flatten_weights(population[0]))

    flat_population = [flatten_weights(ind) for ind in population]

    for i in range(len(population)):
        # Seleciona 3 indivíduos distintos
        indices = list(range(len(population)))
        indices.remove(i)
        a, b, c = random.sample(indices, 3)

        # Mutação
        mutant = flat_population[a] + F * (flat_population[b] - flat_population[c])

        # Crossover
        target = flat_population[i]
        trial = np.copy(target)
        for j in range(dim):
            if random.random() < CR:
                trial[j] = mutant[j]

        # Avaliação
        trial_structured = structure_weights(trial, base_model)
        set_weights(base_model, trial_structured)
        trial_fitness = evaluate_fitness(trial_structured, scenario, base_model, view=False)

        # Seleção
        if trial_fitness >= fitnesses[i]:
            new_population.append(trial_structured)
        else:
            new_population.append(population[i])

    return new_population


# --- Função para obter as dimensões de entrada/saída do ambiente
def get_input_output_sizes(scenario):
    """Obtém as dimensões de entrada/saída do ambiente"""
    robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
    ])
    
    connectivity = get_full_connectivity(robot_structure)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    
    # Obter dimensões de entrada e saída
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    
    env.close()
    
    print(f"Input Size: {input_size}, Output Size: {output_size}")
    
    return input_size, output_size

# --- Função principal para executar o algoritmo ---
def run_de(seed, scenario):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    input_size, output_size = get_input_output_sizes(scenario)
    population, base_model = initialize_controller_population(input_size, output_size, size=POP_SIZE)

    best_global_fitness = float('-inf')
    best_global_solution = None
    start = time.time()

    for generation in range(NUM_GENERATIONS):
        fitnesses = []
        for individual_weights in population:
            set_weights(base_model, individual_weights)
            fitness = evaluate_fitness(individual_weights, scenario, base_model, view=False)
            fitnesses.append(fitness)

            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_solution = flatten_weights(individual_weights)

        population = evolve_controllers(population, scenario, fitnesses, base_model)
        
        parameters = {
            "algorithm": "Differential Evolution",
            "population_size": POP_SIZE,
            "num_generations": NUM_GENERATIONS,
            "scenario": scenario,
            "steps": STEPS,
            "seed": seed,
            "controller_name": "NeuralController"
        }
        
        # Salvar os dados da geração
        save_generation_data(generation, population, fitnesses, scenario, "NeuralController", seed, parameters)
        print(f"Geração {generation + 1}/{NUM_GENERATIONS} | Best fitness: {best_global_fitness:.2f}")

    best_weights = structure_weights(best_global_solution, base_model)
    set_weights(base_model, best_weights)

    execution_time = time.time() - start
    print(f"Tempo total: {execution_time:.2f} segundos")
    print(f"Best fitness: {best_global_fitness:.2f}")

    controller_weights = flatten_weights(get_weights(base_model))
 
    # Salvar os resultados num Excel
    save_results_to_excel(
        controller=NeuralController,
        best_fitness=best_global_fitness,
        scenario=scenario,
        population_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        execution_time=execution_time,
        seed=seed,
        controller_weights=controller_weights,
        filename='task3_2_Results_Complete.xlsx'
    )

    for _ in range(10):
        visualize_policy(best_weights, scenario, base_model)

    utils.create_gif_to_task3_2(robot_structure, filename=f'DE_{scenario}_seed{seed}.gif', scenario=scenario, steps=STEPS, controller=base_model)


# --- Visualização ---
def visualize_policy(weights, scenario, brain):
    set_weights(brain, weights)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]

    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()
    

# --- Função principal ---
def main():
    mode = input("Selecione o modo (1=Run único, 2=5 execuções por cenário): ")

    if mode == '1':
        num = input("Escolha o cenário pelo número (1: DownStepper-v0 ou 2: ObstacleTraverser-v0): ")
        
        if num == '1':
            scenario = 'DownStepper-v0'
        elif num == '2':
            scenario = 'ObstacleTraverser-v0'
        else:
            print("Cenário inválido.")
            return
        
        seed = int(input("Seed a utilizar: "))
        run_de(seed, scenario)

    elif mode == '2':
       for scenario in SCENARIOS:
           for seed in SEEDS:
                print(f"\n--- Executando DE para o cenário {scenario} com a seed {seed} ---")
                run_de(seed, scenario)
    else:
        print("Modo inválido.")

if __name__ == '__main__':
    main()

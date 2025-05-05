
import numpy as np
import pandas as pd
import random
import gymnasium as gym
from evogym.envs import *    
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import utils
import cma
import torch
import os
import csv
from datetime import datetime
from joblib import Parallel, delayed

# === PARÂMETROS ===
NUM_GENERATIONS = 10
STEPS = 500
SCENARIOS = ['DownStepper-v0', 'ObstacleTraverser-v0']
SEEDS = [46, 47, 48, 49, 50]


# === DEFINIÇÃO DO ROBÔ ===
robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])
connectivity = get_full_connectivity(robot_structure)

# === FUNÇÕES AUXILIARES ===
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def unflatten_weights(flat_weights, model):
    new_weights = []
    pointer = 0
    for param in model.parameters():
        shape = param.data.shape
        size = param.data.numel()
        segment = flat_weights[pointer:pointer + size]
        new_weights.append(segment.reshape(shape))
        pointer += size
    return new_weights

def save_generation_data(generation, population, fitness_scores, scenario, controller_name, seed, parameters):
    """
    Salva os dados de cada geração em um arquivo CSV
    """
    folder = f"results_seed_{seed}/{controller_name}_{scenario}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Cabeçalho com metadados da run (apenas na primeira geração)
        if generation == 0:
            writer.writerow(["# ALGORITHM", parameters.get("algorithm", "CMA-ES")])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        # Cabeçalho dos dados por indivíduo
        writer.writerow(["Index", "Fitness", "Reward", "Weights"])

        for i, (weights, fitness) in enumerate(zip(population, fitness_scores)):
            weights_str = str(weights)  # Converter pesos para string para armazenar no CSV
            writer.writerow([i, -fitness, fitness, weights_str])
            
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
        'Algorithm': ["CMA-ES"],
        'Controller Weights': [weights_str]  # Adicionando os pesos e bias
    }

    new_df = pd.DataFrame(new_data)

    # Se o arquivo já existe, adiciona os novos dados
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_excel(filename, index=False)
    print(f"Resultados salvos em {filename}")


# === FUNÇÃO DE AVALIAÇÃO ===
def evaluate_fitness(flat_weights,scenario, brain, view=False):
    weights = unflatten_weights(flat_weights, brain)
    set_weights(brain, weights)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    state = env.reset()[0]
    total_reward = 0
    
    #Adicionar penaliações de queda e movimento para trás

    for t in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()

        if view:
            viewer.render('screen')

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    viewer.close()
    env.close()

    return -total_reward  # CMA-ES minimiza, por isso usamos negativo

def run_cma(seed, scenario):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # === AMBIENTE E REDE ===
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)

    # === INICIALIZA CMA-ES ===
    initial_weights_structured = get_weights(brain)
    initial_weights = flatten_weights(initial_weights_structured)
    
    es = cma.CMAEvolutionStrategy(initial_weights, 0.3, {'popsize': 50, 'seed': seed}) #População de 50
    
    best_global_fitness = float('inf')
    best_global_solution = None

    start = time.time()
    for generation in range(NUM_GENERATIONS):
        solutions = es.ask()

        fitnesses = [evaluate_fitness(sol, scenario, brain) for sol in solutions]

        es.tell(solutions, fitnesses)
        
        if -min(fitnesses) < best_global_fitness:
            best_global_fitness = -min(fitnesses)
            best_global_solution = solutions[fitnesses.index(min(fitnesses))]

        # SALVAR RESULTADOS
        parameters = {
            "algorithm": "CMA-ES",
            "population_size": 50,
            "num_generations": NUM_GENERATIONS,
            "scenario": scenario,
            "steps": STEPS,
            "seed": seed,
            "controller_name": "NeuralController"
        }

        save_generation_data(
            generation=generation,
            population=solutions,
            fitness_scores=fitnesses,
            scenario=scenario,
            controller_name="NeuralController",
            seed=seed,
            parameters=parameters
        )

        print(f"Generation {generation + 1}/{NUM_GENERATIONS} | Best fitness: {-min(fitnesses):.2f}")

    best_weights = unflatten_weights(best_global_solution, brain)
    set_weights(brain, best_weights)
    execution_time = time.time() - start

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Best fitness: {best_global_fitness:.2f}")

    controller_weights = flatten_weights(get_weights(brain))

    save_results_to_excel(
        controller=NeuralController,
        best_fitness=best_global_fitness,
        scenario=scenario,
        population_size=50,
        num_generations=NUM_GENERATIONS,
        execution_time=execution_time,
        seed=seed,
        controller_weights=controller_weights
    )

    for _ in range(10):
        visualize_policy(best_weights, scenario, brain)

    utils.create_gif(robot_structure, filename=f'CMA-ES_{scenario}_seed{seed}.gif', scenario=scenario, steps=STEPS, controller=brain)


# === VISUALIZAÇÃO FINAL ===
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
    

#Acrescentar o modo de 5 execuções de cada 100 runs
# === EXECUÇÃO PRINCIPAL ===
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
        run_cma(seed, scenario)

    elif mode == '2':
       for scenario in SCENARIOS:
           for seed in SEEDS:
                print(f"\n--- Executando CMA-ES para o cenário {scenario} com a seed {seed} ---")
                run_cma(seed, scenario)
    else:
        print("Modo inválido.")

if __name__ == '__main__':
    main()

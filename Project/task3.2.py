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

# === PARÂMETROS ===
NUM_GENERATIONS = 100
STEPS = 500
SCENARIO = 'DownStepper-v0'
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# === DEFINIÇÃO DO ROBÔ ===
robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])
connectivity = get_full_connectivity(robot_structure)

# === AMBIENTE E REDE ===
env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
brain = NeuralController(input_size, output_size)


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

def is_connected(structure):
    # Placeholder — substitui por uma função real se tiveres uma
    return True  # ou verifica com alguma função do EvoGym

def save_generation_data(generation, population, fitness_scores, scenario, controller_name, seed, parameters):
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
            
def save_results_to_excel(controller, best_fitness, scenario, population_size, num_generations, execution_time, seed, controller_weights, filename='task3_1_Results_Complete.xlsx'):
    """
    Salva os resultados em um arquivo Excel, incluindo os pesos e bias do controlador.
    """
    # Converter os pesos e bias para uma string (ou você pode usar um formato mais sofisticado, como JSON)
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
def evaluate_fitness(flat_weights, view=False):
    weights = unflatten_weights(flat_weights, brain)
    set_weights(brain, weights)

    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    state = env.reset()[0]
    total_reward = 0

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


# === INICIALIZA CMA-ES ===
initial_weights_structured = get_weights(brain)
initial_weights = flatten_weights(initial_weights_structured)

es = cma.CMAEvolutionStrategy(initial_weights, 0.5, {'popsize': 20, 'seed': SEED}) #popsize é o número de soluções a gerar por geração # o sigma é o desvio padrão da distribuição normal


# === EVOLUÇÃO ===
start = time.time()
for generation in range(NUM_GENERATIONS):
    solutions = es.ask()
    fitnesses = [evaluate_fitness(sol) for sol in solutions]
    es.tell(solutions, fitnesses)
    
    parameters = {
    "algorithm": "CMA-ES",
    "population_size": 20,
    "num_generations": NUM_GENERATIONS,
    "scenario": SCENARIO,
    "steps": STEPS,
    "seed": SEED,
    "controller_name": "NeuralController"
    }

    save_generation_data(
    generation=generation,
    population=solutions,  # Aqui você passa a população de indivíduos (pesos do controlador)
    fitness_scores=fitnesses,  # Lista de fitnesses
    scenario=SCENARIO,
    controller_name="NeuralController",
    seed=SEED,
    parameters=parameters
    )

    best_gen_fitness = -min(fitnesses)
    print(f"Generation {generation + 1}/{NUM_GENERATIONS} | Best fitness: {best_gen_fitness:.2f}")

best_flat_weights = es.result.xbest
best_weights = unflatten_weights(best_flat_weights, brain)
set_weights(brain, best_weights)

end = time.time()
execution_time = end - start
print(f"Execution time: {execution_time:.2f} seconds")
print(f"Best fitness: {-evaluate_fitness(best_flat_weights):.2f}")
print(f"Best weights: {best_flat_weights}")


# === VISUALIZAÇÃO FINAL ===
def visualize_policy(weights):
    set_weights(brain, weights)

    env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
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
    
# Obter os pesos do controlador (seu modelo neural)
controller_weights = flatten_weights(get_weights(brain))  # Ou qualquer outra função que extraia os pesos

# Agora, chame a função de salvamento passando os pesos
save_results_to_excel(
    controller=NeuralController, 
    best_fitness=-evaluate_fitness(best_flat_weights), 
    scenario=SCENARIO, 
    population_size=20, 
    num_generations=NUM_GENERATIONS, 
    execution_time=execution_time, 
    seed=SEED,
    controller_weights=controller_weights  # Passando os pesos e bias para a função
)


# Reproduz o melhor 10x
for _ in range(10):
    visualize_policy(best_weights)


# Cria gif do melhor comportamento
utils.create_gif(robot_structure, filename='best_Controller.gif', scenario=SCENARIO, steps=STEPS, controller=brain)

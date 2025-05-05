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
import time
from datetime import datetime

# === PARÂMETROS ===
NUM_GENERATIONS = 10
POP_SIZE = 50
STEPS = 500
SEEDS = [42, 43, 44, 45, 46]
SCENARIOS = ['DownStepper-v0', 'ObstacleTraverser-v0']
FM = 0.8   # Fator de mutação
CR = 0.9  # Crossover rate

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
    folder = f"results_seed_{seed}/{controller_name}_{scenario}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if generation == 0:
            writer.writerow(["# ALGORITHM", parameters.get("algorithm", "DE")])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        writer.writerow(["Index", "Fitness", "Reward", "Weights"])
        for i, (weights, fitness) in enumerate(zip(population, fitness_scores)):
            weights_str = str(weights)
            writer.writerow([i, -fitness, fitness, weights_str])

def save_results_to_excel(controller, best_fitness, scenario, population_size, num_generations, execution_time, seed, controller_weights, filename='task3_2_Results_Complete.xlsx'):
    weights_str = str(controller_weights)
    new_data = {
        'Scenario': [scenario],
        'Controller': [controller.__name__],
        'Population Size': [population_size],
        'Number of Generations': [num_generations],
        'Mutation Factor (F)': [FM],
        'Crossover Rate (CR)': [CR],
        'Best Fitness': [best_fitness],
        'Execution Time (s)': [execution_time],
        'Seed': [seed],
        'Algorithm': ["Differential Evolution"],
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

# === FUNÇÃO DE AVALIAÇÃO ===
def evaluate_fitness(flat_weights, scenario, brain, view=False):
    weights = unflatten_weights(flat_weights, brain)
    set_weights(brain, weights)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    state = env.reset()[0]
    total_reward = 0
       
    action_size = sim.get_dim_action_space('robot') 

    initial_pos = sim.object_pos_at_time(0, 'robot')
    positions = [initial_pos]
    max_distance = 0
    stability_penalty = 0
    energy_usage = 0
    backward_steps = 0
    stuck_steps = 0  # Novo contador de passos parados
    
    for t in range(STEPS):  
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        
        if view:
            viewer.render('screen')

        ob, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        current_pos = sim.object_pos_at_time(t, 'robot')
        positions.append(current_pos)

        distance_traveled = np.mean(current_pos[0]) - np.mean(initial_pos[0])
        max_distance = max(max_distance, distance_traveled)

        if len(positions) > 1:
            delta = np.mean(current_pos[0]) - np.mean(positions[-2][0])
            if delta < -0.001:
                backward_steps += 1
            elif abs(delta) < 0.001:
                stuck_steps += 1

        orientation = sim.object_orientation_at_time(t, 'robot')
        adjusted_orientation = (orientation + np.pi) % (2 * np.pi) - np.pi
        if abs(adjusted_orientation) > 0.5:
            stability_penalty += 0.5

        energy_usage += float(np.sum(np.abs(action)))

        if terminated or truncated:
            break
        
    viewer.close()
    env.close()
        
    # Limiares por cenário
    if scenario == 'Walker-v0':
        MAX_THEORETICAL_DISTANCE = STEPS * 0.1  
        MAX_THEORETICAL_REWARD = STEPS * 0.1
    elif scenario == 'BridgeWalker-v0':
        MAX_THEORETICAL_DISTANCE = STEPS * 0.08 
        MAX_THEORETICAL_REWARD = STEPS * 0.08
    else:
        MAX_THEORETICAL_DISTANCE = STEPS * 0.1
        MAX_THEORETICAL_REWARD = STEPS * 0.1
            
    MAX_STABILITY_PENALTY = STEPS * 0.5
    MAX_ENERGY_USAGE = STEPS * action_size * 2
    MAX_STUCK_STEPS = STEPS * 0.3  # Considera 30% parado como um mau sinal
        
    norm_distance = min(max_distance / MAX_THEORETICAL_DISTANCE, 1.0)
    norm_reward = min(total_reward / MAX_THEORETICAL_REWARD, 1.0)
    norm_stability = min(stability_penalty / MAX_STABILITY_PENALTY, 1.0)
    norm_energy = min(energy_usage / MAX_ENERGY_USAGE, 1.0)
    norm_stuck = min(stuck_steps / MAX_STUCK_STEPS, 1.0)

    if scenario == 'DownStepper-v0':
        backward_penalty = min(backward_steps * 0.1, 1.0)
        fitness = (
            norm_distance * 2.0 +  # Peso maior para a distância (descer degraus)
            norm_reward * 1.0 -  # Considerando recompensa por movimento bem-sucedido
            norm_stability * 1.0 -  # Penalização maior por instabilidade, pois o cenário envolve degraus
            backward_penalty * 0.2 -  # Penalizar movimento para trás, mas com peso moderado
            norm_energy * 0.2 - # Menor penalização de energia, pois o movimento envolve subidas/descidas
            norm_stuck * 0.5 
        )

    elif scenario == 'ObstacleTraverser-v0':
        backward_penalty = min(backward_steps * 0.2, 2.0)
        fitness = (
            norm_distance * 2.0 +
            norm_reward * 1.0 -
            norm_stability * 1.0 - 
            backward_penalty * 1.0 -
            norm_energy * 0.2 -
            norm_stuck * 0.5 
        )

    return -fitness # Because DE minimizes the fitness function, we return the negative value to maximize it.

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def run_de(seed, scenario):
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    brain = NeuralController(input_size, output_size)

    initial_weights_structured = get_weights(brain)
    initial_weights = flatten_weights(initial_weights_structured)
    DIM = len(initial_weights)

    # === Limites dos pesos (ex: todos [-1, 1]) ===
    bounds = [(-1.0, 1.0)] * DIM

    # === Função de fitness parcial ===
    from functools import partial
    fitness_function = partial(evaluate_fitness, scenario=scenario, brain=brain, view=False)

    # === Executar DE ===
    start = time.time()
    best = None
    best_fitness = float('inf')
    

    for generation, (candidate, fitness) in enumerate(de(fitness_function, bounds, mut=FM, crossp=CR, popsize=POP_SIZE, its=NUM_GENERATIONS)): 
        
        print(f"Generation {generation+1}/{NUM_GENERATIONS} | Best fitness: {fitness:.2f}")

        # Salvar dados da geração (a população não é salva individualmente aqui, mas pode ser modificada se necessário)
        parameters = {
            "algorithm": "Differential Evolution",
            "population_size": POP_SIZE,
            "num_generations": NUM_GENERATIONS,
            "mutation_factor_F": FM,
            "crossover_rate_CR": CR,
            "scenario": scenario,
            "steps": STEPS,
            "seed": seed,
            "controller_name": "NeuralController"
        }

        # Apenas salvamos o melhor indivíduo por geração aqui
        save_generation_data(generation, [candidate], [fitness], scenario, "NeuralController", seed, parameters)

        if fitness < best_fitness:
            best_fitness = fitness
            best = candidate

    execution_time = time.time() - start

    # === Aplicar pesos finais no cérebro ===
    best_weights = unflatten_weights(best, brain)
    set_weights(brain, best_weights)

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Best fitness: {-best_fitness:.2f}")

    controller_weights = flatten_weights(get_weights(brain))
    save_results_to_excel(
        controller=NeuralController,
        best_fitness=-best_fitness,
        scenario=scenario,
        population_size=POP_SIZE,
        num_generations=NUM_GENERATIONS,
        execution_time=execution_time,
        seed=seed,
        controller_weights=controller_weights
    )

    for _ in range(10):
        visualize_policy(best_weights, scenario, brain)

    utils.create_gif(robot_structure, filename=f'DE_{scenario}_seed{seed}.gif', scenario=scenario, steps=STEPS, controller=brain)


# === VISUALIZAÇÃO ===
def visualize_policy(weights, scenario, brain):
    set_weights(brain, weights)

    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    state = env.reset()[0]

    for _ in range(STEPS):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        viewer.render('screen')
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()


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
        run_de(seed, scenario)

    elif mode == '2':
       for scenario in SCENARIOS:
           for seed in SEEDS:
               print(f"\n--- Executando CMA-ES para o cenário {scenario} com a seed {seed} ---")
               run_de(seed, scenario)
    else:
        print("Modo inválido.")

if __name__ == '__main__':
    main()

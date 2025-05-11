import copy
import numpy as np
import random
import gymnasium as gym
import cma
import torch
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, is_connected, sample_robot
from neural_controller import NeuralController, get_weights, set_weights
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import utils
import time
import json
from concurrent.futures import ProcessPoolExecutor

# ----- PARÂMETROS GERAIS -----
NUM_GENERATIONS = 50
POPULATION_SIZE = 20
STEPS = 500
SCENARIO = 'CaveCrawler-v0' 
SEED = 42
ROBOT_SIZE = (5, 5)

# ----- PARÂMETROS PARA GA (ESTRUTURAS) -----
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3

# ----- PARÂMETROS PARA CMA-ES (CONTROLADORES) -----
SIGMA_INIT = 0.2

# ----- PARÂMETROS PARA AVALIAÇÃO CRUZADA -----
NUM_PARTNERS = 3  # Número de parceiros para avaliação

# ----- VOXEL TYPES -----
# 0: Empty, 1: Rigid, 2: Soft, 3: Horizontal Actuator, 4: Vertical Actuator
VOXEL_TYPES = [0, 1, 2, 3, 4]

# ----- ALGORITMO -----
ALGORITHM_NAME = "COEVOLUTION-GA-CMAES"  

# ----- CONFIGURAÇÃO INICIAL -----
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def save_generation_data(generation, structures, controllers, fitness_matrix, best_combination, best_fitness,
                         scenario, seed, parameters):
  
    folder = f"results_seed_{seed}/GA_CMAES_2{scenario}"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        if generation == 0:
            writer.writerow(["# ALGORITHM", ALGORITHM_NAME])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        writer.writerow([f"# BEST_COMBINATION_STRUCTURE", best_combination[0]])
        writer.writerow([f"# BEST_COMBINATION_CONTROLLER", best_combination[1]])
        writer.writerow([f"# BEST_FITNESS", best_fitness])
        
        writer.writerow([
            "Structure_Index", "Controller_Index", "Fitness", 
            "Connected", "Structure", "Controller_Weights"
        ])

        for i, structure in enumerate(structures):
            for j, controller in enumerate(controllers):
                fitness = fitness_matrix[i, j]
                connected = is_connected(structure)
                structure_flat = structure.flatten().tolist()
                controller_weights = json.dumps([w.tolist() for w in controller])  
                writer.writerow([
                    i, j, fitness, connected, structure_flat, controller_weights
                ])

def save_results_to_excel(best_fitness, best_robot, best_controller, scenario, 
                          population_size, num_generations, mutation_rate, crossover_rate, 
                          tournament_size, execution_time, seed, filename='task3_3_Results.xlsx'):
  
    controller_summary = f"Neural network with {len(best_controller)} weight matrices"
    
    controller_weights_json = json.dumps([w.tolist() for w in best_controller])

    new_data = {
        'Scenario': [scenario],
        'Population Size': [population_size],
        'Number of Generations': [num_generations],
        'Best Fitness': [best_fitness],
        'Best Robot Structure': [str(best_robot)],
        'Best Controller Summary': [controller_summary],
        'Best Controller Weights': [controller_weights_json],  
        'Mutation Rate': [mutation_rate],
        'Crossover Rate': [crossover_rate],
        'Tournament Size': [tournament_size],
        'Execution Time (s)': [execution_time],
        'Seed': [seed],
        'Algorithm': [ALGORITHM_NAME]
    }

    new_df = pd.DataFrame(new_data)

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")
    
def save_history(history, filename):
    
    df = pd.DataFrame(history)
    
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df
    
    combined_df.to_csv(filename, index=False)
    print(f"Histórico salvo em {filename}")

# ----- FUNÇÕES PARA A EVOLUÇÃO DA ESTRUTURA (GA) -----

def initialize_structure_population(size=POPULATION_SIZE):
    population = []
    for _ in range(size):
        robot_structure, _ = sample_robot(ROBOT_SIZE)
        population.append(robot_structure)
    return population

def mutate_structure(structure):
    for attempt in range(10):  
        mutated = structure.copy()

        for i in range(ROBOT_SIZE[0]):  
            for j in range(ROBOT_SIZE[1]): 
                if random.random() < MUTATION_RATE:
                    # Escolhemos um novo tipo de voxel aleatoriamente
                    mutated[i, j] = random.choice(VOXEL_TYPES)

        # Garantimos que a estrutura tenha pelo menos um voxel
        if np.sum(mutated) == 0:
            i, j = random.randint(0, ROBOT_SIZE[0] - 1), random.randint(0, ROBOT_SIZE[1] - 1)
            mutated[i, j] = random.choice([1, 2, 3, 4])

        # Verificar conectividade
        if is_connected(mutated):
            return mutated
        
    return structure  


def mutate_diverse(robot_structure, generation, max_attempts=10):
    for attempt in range(max_attempts):
        mutated = copy.deepcopy(robot_structure)

        # Escolha baseada na fase da evolução
        phase = generation / NUM_GENERATIONS

        if phase < 0.3:  # Fase inicial: exploração ampla
            mutation_probs = {"point": 0.1, "block": 0.2, "add_remove": 0.2, "add_actuator": 0.2, "add_jump_leg": 0.3}
        elif phase < 0.7:  # Fase média: equilíbrio
            mutation_probs = {"point": 0.3, "block": 0.2, "add_remove": 0.2, "add_actuator": 0.2, "add_jump_leg": 0.1}
        else:  # Fase final: refinamento
            mutation_probs = {"point": 0.6, "block": 0.1, "add_remove": 0.1, "add_actuator": 0.1, "add_jump_leg": 0.1}

        mutation_type = random.choices(
            list(mutation_probs.keys()),
            weights=list(mutation_probs.values())
        )[0]

        if mutation_type == "point":
            for i in range(5):
                for j in range(5):
                    if random.random() < MUTATION_RATE / 5:
                        mutated[i, j] = random.choice(VOXEL_TYPES)

        elif mutation_type == "block":
            if random.random() < MUTATION_RATE:
                start_i = random.randint(0, 3)
                start_j = random.randint(0, 3)
                block_type = random.choice(VOXEL_TYPES)
                mutated[start_i:start_i+2, start_j:start_j+2] = block_type

        elif mutation_type == "add_remove":
            i, j = random.randint(0, 4), random.randint(0, 4)
            if mutated[i, j] == 0:
                mutated[i, j] = random.choice([1, 2, 3, 4])
            else:
                if random.random() < 0.5:
                    if mutated[i, j] in [3, 4] and np.sum(np.isin(mutated, [3, 4])) <= 1:
                        pass
                    else:
                        mutated[i, j] = 0

        elif mutation_type == "add_actuator":
            i, j = random.randint(0, 4), random.randint(0, 4)
            mutated[i, j] = random.choice([3, 4])

        elif mutation_type == "add_jump_leg":
            col = random.randint(0, 4)
            height = random.randint(2, 4)
            for row in range(5 - height, 5):
                mutated[row, col] = 1  # material sólido
            mutated[4, col] = random.choice([3, 4])  # atuador na base

        # Forçar simetria horizontal para melhorar equilíbrio
        mutated = np.maximum(mutated, np.fliplr(mutated))

        # Garante pelo menos um atuador
        if not np.any(np.isin(mutated, [3, 4])):
            i, j = random.randint(0, 4), random.randint(0, 4)
            mutated[i, j] = random.choice([3, 4])

        # Garante ao menos um atuador nos cantos inferiores
        if mutated[4, 0] not in [3, 4] and mutated[4, 4] not in [3, 4]:
            mutated[4, random.choice([0, 4])] = random.choice([3, 4])

        if is_connected(mutated):
            return mutated, mutation_type

    return copy.deepcopy(robot_structure), "none"


def crossover_structures(parent1, parent2):

    for attempt in range(10):  
        if random.random() > CROSSOVER_RATE:
            return parent1.copy()

        child = np.zeros((ROBOT_SIZE[0], ROBOT_SIZE[1]), dtype=int)

        # Crossover uniforme
        for i in range(ROBOT_SIZE[0]):  
            for j in range(ROBOT_SIZE[1]):  
                if random.random() < 0.5:
                    child[i, j] = parent1[i, j]
                else:
                    child[i, j] = parent2[i, j]

        # Garantimos que a estrutura tenha pelo menos um voxel
        if np.sum(child) == 0:
            i, j = random.randint(0, ROBOT_SIZE[0] - 1), random.randint(0, ROBOT_SIZE[1] - 1)
            child[i, j] = random.choice([1, 2, 3, 4])

        if is_connected(child):
            return child

    return parent1.copy()  

def tournament_selection(population, fitness_values, tournament_size=TOURNAMENT_SIZE):
   
    selected = []
    for _ in range(2):  
        tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])  
        selected.append(winner[0])
    return selected[0], selected[1]

def evolve_structures(structure_population, fitness_values, elitism=2):
    
    # Ordenar a população com base no fitness
    sorted_population = [x for _, x in sorted(zip(fitness_values, structure_population), key=lambda pair: pair[0], reverse=True)]
    
    # Preservar os melhores indivíduos (elitismo)
    new_structures = sorted_population[:elitism]
    
    # Gerar o restante da nova população
    while len(new_structures) < len(structure_population):
        # Seleção
        parent1, parent2 = tournament_selection(structure_population, fitness_values)
        
        # Crossover
        child = crossover_structures(parent1, parent2)
        
        # Mutação
        #child = mutate_structure(child)
        child, mutation_type = mutate_diverse(child, generation=0)  
    
        
        new_structures.append(child)
        
    return new_structures

# ----- FUNÇÕES PARA A EVOLUÇÃO DO CONTROLADOR (CMA-ES) -----

def initialize_controller_population(input_size, output_size, size=POPULATION_SIZE):
    """Inicializa a população de controladores usando CMA-ES"""
    # Criamos um modelo base para obter as dimensões dos pesos
    base_model = NeuralController(input_size, output_size)
    
    # Obtemos o número total de parâmetros para o CMA-ES
    flat_weights = flatten_weights(get_weights(base_model))
    cma_es = cma.CMAEvolutionStrategy(
        flat_weights, 
        SIGMA_INIT,  # Desvio padrão inicial
        {'popsize': size, 'seed':SEED}
    )
    
    # Gera a população inicial de controladores
    cma_solutions = cma_es.ask()
    controller_population = []
    
    for solution in cma_solutions:
        structured_weights = structure_weights(solution, base_model)
        controller_population.append(structured_weights)
        
    return controller_population, cma_es, base_model

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

def evolve_controllers(controller_population, cma_es, base_model, fitness_values):
    """Evolui a população de controladores usando CMA-ES"""
    # Substituir valores não finitos por um valor padrão
    fitness_values = np.array(fitness_values)
    fitness_values[~np.isfinite(fitness_values)] = -100.0

    # Passa os dados de fitness para o CMA-ES (CMA-ES minimiza, então invertemos)
    cma_es.tell([flatten_weights(controller) for controller in controller_population], 
                [-fitness for fitness in fitness_values])
    
    # Gera a nova população
    cma_solutions = cma_es.ask()
    new_controllers = []
    
    for solution in cma_solutions:
        structured_weights = structure_weights(solution, base_model)
        new_controllers.append(structured_weights)
        
    return new_controllers

# ----- FUNÇÕES DE AVALIAÇÃO -----

def get_input_output_sizes(scenario=SCENARIO):
    """Obtém as dimensões de entrada/saída do ambiente"""
    robot_structure = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
    ])
    # Cria uma estrutura mínima para inicializar o ambiente
    connectivity = get_full_connectivity(robot_structure)
    
    # Criar o ambiente
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
    
    # Obter dimensões de entrada e saída
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    env.close()
    
    print(f"Input Size: {input_size}, Output Size: {output_size}")
    
    return input_size, output_size

# https://paperswithcode.com/method/xavier-initialization - Inicialização Xavier
def adapt_weights(old_weights, old_input, old_output, new_input, new_output):
    """
    Adapta os pesos de um controlador com dimensões antigas para novas dimensões.
    - Mantém o máximo possível dos pesos originais.
    - Usa inicialização Xavier para novos valores.
    """
    # Desempacotar os pesos antigos
    old_fc1_w, old_fc1_b, old_fc2_w, old_fc2_b = old_weights

    # Nova camada 1 (fc1): [16, new_input]
    new_fc1_w = np.zeros((16, new_input), dtype=np.float32)
    new_fc1_b = np.zeros((16,), dtype=np.float32)

    # Copiar valores compartilhados para fc1
    min_in = min(old_input, new_input)
    new_fc1_w[:, :min_in] = old_fc1_w[:, :min_in]
    new_fc1_b[:] = old_fc1_b  # Bias permanece o mesmo tamanho

    # ----- INICIALIZAÇÃO XAVIER PARA CAMADA 1 -----
    # A inicialização Xavier resolve o problema de gradientes que desaparecem ou explodem
    # distribuindo os pesos iniciais de forma que a variância dos sinais permaneça constante
    # através das camadas da rede neural.
    if new_input > old_input:
        # fan_in = número de neurônios de entrada conectados a cada neurônio desta camada
        fan_in = new_input  
        
        # Cálculo do limite para distribuição uniforme usando a fórmula Xavier:
        # sqrt(6/(fan_in + fan_out)) onde fan_out=16 neste caso (número de neurônios nesta camada)
        # Isto garante que a variância dos pesos será aproximadamente 2/(fan_in + fan_out)
        bound = np.sqrt(6. / (fan_in + 16))
        
        # Gera pesos aleatórios dentro do intervalo [-bound, bound]
        # Usamos distribuição uniforme para garantir que os valores estejam bem distribuídos
        # Os pesos são inicializados apenas para as novas conexões (old_input até new_input)
        new_fc1_w[:, old_input:] = np.random.uniform(-bound, bound, size=(16, new_input - old_input))
        
        # Resultado: Os sinais de ativação não ficam nem muito grandes nem muito pequenos quando
        # atravessam a rede, facilitando o treinamento e convergência

    # Nova camada 2 (fc2): [new_output, 16]
    new_fc2_w = np.zeros((new_output, 16), dtype=np.float32)
    new_fc2_b = np.zeros((new_output,), dtype=np.float32)

    # Copiar valores compartilhados para fc2
    min_out = min(old_output, new_output)
    new_fc2_w[:min_out, :] = old_fc2_w[:min_out, :]
    new_fc2_b[:min_out] = old_fc2_b[:min_out]

    # ----- INICIALIZAÇÃO XAVIER PARA CAMADA 2 -----
    if new_output > old_output:
        # fan_in = número de neurônios na camada anterior (16)
        fan_in = 16
        
        # fan_out = número de neurônios na camada de saída
        fan_out = new_output
        
        # Mesmo princípio da inicialização Xavier: mantém a variância do sinal
        # constante através das camadas da rede
        bound = np.sqrt(6. / (fan_in + fan_out))
        
        # Inicializa pesos entre [-bound, bound] para as novas conexões de saída
        new_fc2_w[old_output:, :] = np.random.uniform(-bound, bound, size=(new_output - old_output, 16))
        
        # Também inicializamos os bias usando a mesma distribuição
        new_fc2_b[old_output:] = np.random.uniform(-bound, bound, size=(new_output - old_output,))

    return [new_fc1_w, new_fc1_b, new_fc2_w, new_fc2_b]


def evaluate_fitness(structure, controller_weights, input_size, output_size, view=False, max_attempts=10):
    attempts = 0

    while attempts < max_attempts:
        try:
            if not is_connected(structure):
                return -100.0

            connectivity = get_full_connectivity(structure)
            env = gym.make(SCENARIO, max_episode_steps=STEPS,
                           body=structure, connections=connectivity)

            actual_input_size = env.observation_space.shape[0]
            actual_output_size = env.action_space.shape[0]

            # Adapta os pesos apenas se as dimensões forem diferentes
            if actual_input_size != input_size or actual_output_size != output_size:
                try:
                    new_weights = adapt_weights(
                        controller_weights,
                        old_input=input_size,
                        old_output=output_size,
                        new_input=actual_input_size,
                        new_output=actual_output_size
                    )
                except Exception as e:
                    print(f"Erro ao adaptar pesos: {e}")
                    env.close()
                    attempts += 1
                    continue
            else:
                new_weights = controller_weights

            controller = NeuralController(actual_input_size, actual_output_size)

            try:
                set_weights(controller, new_weights)
            except Exception as e:
                print(f"Erro ao definir pesos no controlador: {e}")
                env.close()
                attempts += 1
                continue

            state = env.reset()[0]
            sim = env.sim

            if view:
                viewer = EvoViewer(sim)
                viewer.track_objects('robot')

            total_reward = 0

            for t in range(STEPS):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = controller(state_tensor).detach().numpy().flatten()

                if view:
                    viewer.render('screen')

                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            if view:
                viewer.close()
            env.close()

            if not np.isfinite(total_reward):
                print(f"Fitness inválido detectado: {total_reward}. Retornando -100.0.")
                return -100.0

            return total_reward

        except (ValueError, IndexError, Exception) as e:

            if 'env' in locals():
                env.close()
            attempts += 1

    return -100.0

def evaluate_combination(args):
    """Função auxiliar para avaliar uma combinação de estrutura e controlador."""
    i, j, structure_population, controller_population, input_size, output_size = args
    return i, j, evaluate_fitness(structure_population[i], controller_population[j], input_size, output_size)

def evaluate_combinations_parallel(structure_population, controller_population, input_size, output_size, full_eval=False, num_samples=100):
    num_structures = len(structure_population)
    num_controllers = len(controller_population)
    
    # Inicializar matriz de fitness com NaN
    fitness_matrix = np.full((num_structures, num_controllers), np.nan)
    
    # Avaliação completa
    if full_eval:
        tasks = [(i, j, structure_population, controller_population, input_size, output_size) 
                 for i in range(num_structures) for j in range(num_controllers)]
    else:
        # Avaliação parcial
        total_combinations = num_structures * num_controllers
        num_samples = min(num_samples, total_combinations)
        selected_indices = np.random.choice(total_combinations, size=num_samples, replace=False)
        tasks = [(idx // num_controllers, idx % num_controllers, structure_population, controller_population, input_size, output_size) 
                 for idx in selected_indices]
    
    # Paralelizar a avaliação
    with ProcessPoolExecutor() as executor:
        results = executor.map(evaluate_combination, tasks)
    
    # Preencher a matriz de fitness com os resultados
    for i, j, fitness in results:
        fitness_matrix[i, j] = fitness
        print(f"Fitness avaliado para estrutura {i} e controlador {j}: {fitness}")
    
    return fitness_matrix


def calculate_structure_fitness(fitness_matrix, i):
    """Calcula a fitness de uma estrutura com base nos K melhores controladores"""

    controller_scores = fitness_matrix[i, :]
    valid_scores = controller_scores[~np.isnan(controller_scores)]
    
    if len(valid_scores) == 0:
        return float('-inf')  # Sem scores válidos
    
    # Ajustar K se necessário
    k = min(NUM_PARTNERS, len(valid_scores))
    
    # Selecionar os K melhores scores
    top_k_scores = np.sort(valid_scores)[-k:]
    
    return np.mean(top_k_scores)

def calculate_controller_fitness(fitness_matrix, j):
    """Calcula a fitness de um controlador com base nas K melhores estruturas"""
    # Ordenamos as estruturas pelo desempenho com este controlador
    structure_scores = fitness_matrix[:,j]
    valid_scores = structure_scores[~np.isnan(structure_scores)]
    
    if len(valid_scores) == 0:
        return float('-inf')  # Sem scores válidos
    
    # Ajustar K se necessário
    k = min(NUM_PARTNERS, len(valid_scores))
    
    # Selecionar os K melhores scores
    top_k_scores = np.sort(valid_scores)[-k:]
    
    return np.mean(top_k_scores)




def run_coevolution(seed, num_generations=NUM_GENERATIONS, eval_full_every=10, samples_per_gen=100):
   
    # Inicialização
    print("Inicializando populações...")
    input_size, output_size = get_input_output_sizes()
    structure_population = initialize_structure_population()
    controller_population, cma_es, base_model = initialize_controller_population(input_size, output_size)
        
 
    best_structure = None
    best_controller = None
    best_fitness = float('-inf')
    best_combination = (0, 0)
    
    history = []
    
    # Matriz de fitness acumulada (para manter resultados de avaliações anteriores)
    # Inicializada com NaN para indicar que nenhuma combinação foi avaliada ainda
    accumulated_fitness = np.full((len(structure_population), len(controller_population)), np.nan)

    start_time = time.time()
    
    # Salvar a geração inicial - sempre com avaliação completa
    print("Salvando a geração inicial...")
    fitness_matrix = evaluate_combinations_parallel(
        structure_population, 
        controller_population, 
        input_size, 
        output_size, 
        full_eval=True
    )
    accumulated_fitness = fitness_matrix.copy()  
    
    parameters = {
        "scenario": SCENARIO,
        "mutation_rate": MUTATION_RATE,
        "crossover_rate": CROSSOVER_RATE,
        "tournament_size": TOURNAMENT_SIZE,
        "sigma_init": SIGMA_INIT,
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "eval_full_every": eval_full_every,
        "samples_per_gen": samples_per_gen,
        "seed": seed
    }
    
    save_generation_data(
        generation=0,
        structures=structure_population,
        controllers=controller_population,
        fitness_matrix=fitness_matrix,
        best_combination=best_combination,
        best_fitness=best_fitness,
        scenario=SCENARIO,
        seed=seed,
        parameters=parameters
    )
  
    for gen in range(1, num_generations + 1):  
        generation_start = time.time()
        
        # 1. Determinar se esta geração deve ter avaliação completa
        full_evaluation = (gen % eval_full_every == 0)
        next_full_evaluation = ((gen + 1) % eval_full_every == 0)
        
        if full_evaluation:
            print(f"Geração {gen}: Realizando avaliação COMPLETA de todas as combinações...")
        else:
            print(f"Geração {gen}: Realizando avaliação PARCIAL com {samples_per_gen} combinações...")
        
        # 2. Avaliação de combinações (completa ou parcial)
        current_fitness = evaluate_combinations_parallel(
            structure_population, 
            controller_population, 
            input_size, 
            output_size, 
            full_eval=full_evaluation,
            num_samples=samples_per_gen
        )
        
        # 3. Atualizar a matriz de fitness acumulada com os novos valores avaliados
        # Substitui apenas os valores que foram avaliados nesta geração
        mask = ~np.isnan(current_fitness)
        accumulated_fitness[mask] = current_fitness[mask]
        
        # 4. Usar a matriz acumulada para cálculos
    
        print(accumulated_fitness)
        
        # 5. Encontrar a melhor combinação
        # Ignorar valores NaN ao encontrar o máximo
        with np.errstate(invalid='ignore'):  # Ignora avisos sobre NaN
            valid_indices = ~np.isnan(accumulated_fitness)
            if np.any(valid_indices):  # Verifica se há pelo menos um valor válido
                valid_max = np.nanmax(accumulated_fitness)
                max_indices = np.where(accumulated_fitness == valid_max)
                best_i, best_j = max_indices[0][0], max_indices[1][0]
                current_best_fitness = accumulated_fitness[best_i, best_j]
            else:
                current_best_fitness = float('-inf')
                best_i, best_j = 0, 0
        
        # 6. Atualizar o melhor global se necessário
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_structure = structure_population[best_i].copy()
            best_controller = controller_population[best_j].copy()
            best_combination = (best_i, best_j)
            print(f"Nova melhor combinação! Fitness: {best_fitness:.2f}")
        
        # 7. Calcular fitness para cada estrutura e controlador
        # Usando np.nanmean para ignorar NaN ao calcular médias
        
        # Linhas estruturas 
        # Colunas controladores
        # fitness_matrix[i, j] = fitness da estrutura i com controlador j
        
        structure_fitness = np.array([
            calculate_structure_fitness(accumulated_fitness, i) 
            for i in range(len(structure_population))
        ])
        
        controller_fitness = np.array([
            calculate_controller_fitness(accumulated_fitness, j) 
            for j in range(len(controller_population))
        ])
        
        
        # 8. Evolução das populações
        structure_population = evolve_structures(structure_population, structure_fitness)
        controller_population = evolve_controllers(
            controller_population, 
            cma_es, 
            base_model, 
            controller_fitness
        )
        
 
        generation_time = time.time() - generation_start
        

        history.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': np.nanmean(accumulated_fitness),
            'std_fitness': np.nanstd(accumulated_fitness),
            'time_per_gen': generation_time,
            'full_evaluation': full_evaluation
        })
        

        print(f"Geração {gen}/{num_generations}: Melhor Fitness = {current_best_fitness:.2f}, " + 
              f"Global = {best_fitness:.2f}, Tempo = {generation_time:.2f}s")
        
     
        save_generation_data(
            generation=gen,
            structures=structure_population,
            controllers=controller_population,
            fitness_matrix=accumulated_fitness, 
            best_combination=best_combination,
            best_fitness=best_fitness,
            scenario=SCENARIO,
            seed=seed,
            parameters=parameters
        )
    
 
    total_time = time.time() - start_time
    print(f"\nEvolução completa! Tempo total: {total_time:.2f}s")
    print(f"Melhor fitness global: {best_fitness:.2f}")
    

    save_history(history, f"coevolution_GA_CMAES_{SCENARIO}_seed{seed}_history_2.csv")
    
    save_results_to_excel(
        best_fitness=best_fitness,
        best_robot=best_structure,
        best_controller=best_controller,
        scenario=SCENARIO,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        execution_time=total_time,
        seed=seed
    )
    
    
    utils.create_gif_pair(best_structure, best_controller, filename=f"COEVOLUCIONARY_GA_CMAES_{SCENARIO}_seed{seed}_2.gif",duration=0.066,scenario=SCENARIO,steps=500,
                              INPUT_SIZE = input_size, OUTPUT_SIZE=output_size)
    
    return best_structure, best_controller, best_fitness

def visualize_best(structure, controller, input_size, output_size):

    evaluate_fitness(structure, controller, input_size, output_size, view=True)



def run(num_runs=5):
    """Executa o algoritmo em múltiplos cenários e controladores com diferentes seed."""
    scenarios = ['CaveCrawler-v0', 'GapJumper-v0']  
    controllers = [NeuralController]      
    final_results = []
    
    for scenario in scenarios:
        for controller in controllers:
            global SCENARIO, CONTROLLER
            SCENARIO = scenario
            CONTROLLER = controller
            
            print(f"\n\n=== Executando {controller.__name__} no cenário {scenario} ===\n")
            
            for run_idx in range(num_runs):
                RUN_SEED = 42 + run_idx
                np.random.seed(RUN_SEED)
                random.seed(RUN_SEED)
                torch.manual_seed(RUN_SEED)
                
                print(f"\nExecução {run_idx + 1}/{num_runs} com seed {RUN_SEED}")
                
             
                start_time = time.time()
                best_structure, best_controller, best_fitness = run_coevolution(RUN_SEED)
                end_time = time.time()
                
                # Obter dimensões para visualização
                input_size, output_size = get_input_output_sizes()
                
                # Criar o modelo PyTorch com as dimensões corretas
                best_brain = NeuralController(input_size, output_size)

                # Carregar os pesos no modelo
                set_weights(best_brain, best_controller)

                print("\nSalvando pesos do controlador...")
       
                torch.save(best_brain.state_dict(), f"best_controller_{SCENARIO}_seed{RUN_SEED}_2.pt")
                
        
                print("\nSalvando estrutura do robô...")
                np.save(f"best_structure_{SCENARIO}_seed{RUN_SEED}_2.npy", best_structure)
                
                
               
                final_results.append({
                    'scenario': scenario,
                    'controller': controller.__name__,
                    'seed': RUN_SEED,
                    'best_fitness': best_fitness,
                    'execution_time': round(end_time - start_time, 2)
                })
    

    final_df = pd.DataFrame(final_results)
    final_df.to_excel('coevolution_multiple_scenarios_summary_2.xlsx', index=False)
    

    print("\n=== Resumo Final ===")
    for scenario in scenarios:
        for controller_name in [c.__name__ for c in controllers]:
            subset = final_df[(final_df['scenario'] == scenario) & (final_df['controller'] == controller_name)]
            avg_fitness = subset['best_fitness'].mean()
            std_fitness = subset['best_fitness'].std()
            print(f"{scenario} + {controller_name}: Média de Fitness = {avg_fitness:.2f}, Desvio Padrão = {std_fitness:.2f}")
    
    print("\nResultados salvos em 'coevolution_multiple_scenarios_summary.xlsx'")
    
    
def main():
    print("===== Co-Evolução de Estrutura e Controlador (GA + CMA-ES) =====")
    print("1. Executar uma única vez")
    print("2. Executar com múltiplas seed")
    print("3. Executar com visualização apenas")
    
    choice = input("Escolha uma opção (1-3): ")
    
    if choice == '1':
        
        print("\nConfigurando a seed...")
        RUN_SEED = 40  
        np.random.seed(RUN_SEED)
        random.seed(RUN_SEED)
        torch.manual_seed(RUN_SEED)
        
        print("\nIniciando co-evolução...")
        best_structure, best_controller, best_fitness = run_coevolution()
        
        input_size, output_size = get_input_output_sizes()
        
        print(f"\nMelhor robô encontrado com fitness {best_fitness}:")
        print(best_structure)
        
        # Criar o controlador neural com os pesos aprendidos
        best_brain = NeuralController(input_size, output_size)
        set_weights(best_brain, best_controller)
        
       
        print("\nVisualizando o melhor robô...")
        for _ in range(3):  # Visualizar 3 vezes
            visualize_best(best_structure, best_controller, input_size, output_size)
        
   
   
        print("\nSalvando pesos do controlador...")
        torch.save(best_brain.state_dict(), f"best_controller_{SCENARIO}_seed{RUN_SEED}_2.pt")
       
        print("\nSalvando estrutura do robô...")
        np.save(f"best_structure_{SCENARIO}_seed{RUN_SEED}_2.npy", best_structure)
        
    elif choice == '2':
        run(5)
        
    elif choice == '3':
        structure_file = input("Arquivo da estrutura (.npy): ")
        controller_file = input("Arquivo do controlador (.pt): ")
        
        structure = np.load(structure_file)
        input_size, output_size = get_input_output_sizes()
        
        controller_model = NeuralController(input_size, output_size)
        controller_model.load_state_dict(torch.load(controller_file))
        controller = get_weights(controller_model)
        
        print("Visualizando robô...")
        visualize_best(structure, controller, input_size, output_size)
    
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()
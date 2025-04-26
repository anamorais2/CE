import numpy as np
import random
import gymnasium as gym
import cma
import torch
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity, is_connected, sample_robot
from neural_controller import NeuralController, get_weights, set_weights
# Importações para salvar dados
import pandas as pd
import os
import imageio
import matplotlib.pyplot as plt
import csv
import utils
import time
import json

# ----- PARÂMETROS GERAIS -----
NUM_GENERATIONS = 100
POPULATION_SIZE = 10
STEPS = 500
SCENARIO = 'GapJumper-v0'  # ou 'CaveCrawler-v0'
SEED = 42
ROBOT_SIZE = (5, 5)

# ----- PARÂMETROS PARA GA (ESTRUTURAS) -----
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3

# ----- PARÂMETROS PARA CMA-ES (CONTROLADORES) -----
SIGMA_INIT = 0.5

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
    """
    Guarda os dados de uma geração inteira num CSV separado por seed, incluindo os pesos dos controladores.
    
    Args:
        generation: Número da geração
        structures: Lista de estruturas de robôs na população
        controllers: Lista de controladores na população
        fitness_matrix: Matriz de fitness de todas as combinações estrutura-controlador
        best_combination: Índices da melhor combinação (i, j)
        best_fitness: Melhor fitness global até o momento
        scenario: Nome do cenário usado
        seed: Semente aleatória usada
        parameters: Dicionário com os parâmetros da execução
    """
    folder = f"results_seed_{seed}/GA_CMAES_{scenario}"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Cabeçalho com metadados da run (apenas na primeira geração)
        if generation == 0:
            writer.writerow(["# ALGORITHM", ALGORITHM_NAME])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        # Escrever melhor combinação encontrada
        writer.writerow([f"# BEST_COMBINATION_STRUCTURE", best_combination[0]])
        writer.writerow([f"# BEST_COMBINATION_CONTROLLER", best_combination[1]])
        writer.writerow([f"# BEST_FITNESS", best_fitness])

        # Cabeçalho dos dados por combinação
        writer.writerow([
            "Structure_Index", "Controller_Index", "Fitness", 
            "Connected", "Structure", "Controller_Weights"
        ])

        # Salvando dados de cada combinação
        for i, structure in enumerate(structures):
            for j, controller in enumerate(controllers):
                fitness = fitness_matrix[i, j]
                connected = is_connected(structure)
                structure_flat = structure.flatten().tolist()
                controller_weights = json.dumps([w.tolist() for w in controller])  # Converte os pesos para JSON
                writer.writerow([
                    i, j, fitness, connected, structure_flat, controller_weights
                ])

def save_results_to_excel(best_fitness, best_robot, best_controller, scenario, 
                          population_size, num_generations, mutation_rate, crossover_rate, 
                          tournament_size, execution_time, seed, filename='task3_3_Results.xlsx'):
  
    # Extrair informações do controlador
    controller_summary = f"Neural network with {len(best_controller)} weight matrices"
    
    # Converter os pesos do controlador para JSON
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
    """
    Salva o histórico de evolução em um arquivo CSV
    """
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
    """Inicializa a população de estruturas de robôs"""
    population = []
    for _ in range(size):
        robot_structure, _ = sample_robot(ROBOT_SIZE)
        population.append(robot_structure)
    return population

def mutate_structure(structure):
    """Operador de mutação para a estrutura do robô, garantindo conectividade."""
    for attempt in range(10):  # Tentar até 10 vezes
        mutated = structure.copy()

        for i in range(ROBOT_SIZE[0]):  # Altura
            for j in range(ROBOT_SIZE[1]):  # Largura
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

    print("Mutação falhou em criar uma estrutura conectada após 10 tentativas. Retornando a estrutura original.")
    return structure  # Retorna a estrutura original se não conseguir um robo conectado


def crossover_structures(parent1, parent2):
    """Operador de crossover para estruturas de robôs, garantindo conectividade."""
    for attempt in range(10):  # Tentar até 10 vezes
        if random.random() > CROSSOVER_RATE:
            return parent1.copy()

        child = np.zeros((ROBOT_SIZE[0], ROBOT_SIZE[1]), dtype=int)

        # Crossover uniforme
        for i in range(ROBOT_SIZE[0]):  # Altura
            for j in range(ROBOT_SIZE[1]):  # Largura
                if random.random() < 0.5:
                    child[i, j] = parent1[i, j]
                else:
                    child[i, j] = parent2[i, j]

        # Garantimos que a estrutura tenha pelo menos um voxel
        if np.sum(child) == 0:
            i, j = random.randint(0, ROBOT_SIZE[0] - 1), random.randint(0, ROBOT_SIZE[1] - 1)
            child[i, j] = random.choice([1, 2, 3, 4])

        # Verificar conectividade
        if is_connected(child):
            return child

    print("Crossover falhou em criar uma estrutura conectada após 10 tentativas. Retornando o pai 1.")
    return parent1.copy()  # Retorna o pai 1 se não conseguir um robô conectado

def tournament_selection(population, fitness_values, tournament_size=TOURNAMENT_SIZE):
    """Seleção por torneio para a população de estruturas, retornando dois pais."""
    selected = []
    for _ in range(2):  
        tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])  
        selected.append(winner[0])
    return selected[0], selected[1]

def evolve_structures(structure_population, fitness_values):
    """Evolui a população de estruturas usando algoritmo genético."""
    new_structures = []
    for _ in range(len(structure_population)):
        # Seleção
        parent1, parent2 = tournament_selection(structure_population, fitness_values)
        
        # Crossover
        child = crossover_structures(parent1, parent2)
        
        # Mutação
        child = mutate_structure(child)
        
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
        np.zeros(len(flat_weights)),  # Inicialização com zeros
        SIGMA_INIT,  # Desvio padrão inicial
        {'popsize': size}
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
    # Cria uma estrutura mínima para inicializar o ambiente
    sample_structure = np.zeros(ROBOT_SIZE, dtype=int)
    sample_structure[0, 0] = 1  # Pelo menos um voxel é necessário
    connectivity = get_full_connectivity(sample_structure)
    
    env = gym.make(scenario, max_episode_steps=STEPS, 
                  body=sample_structure, connections=connectivity)
    
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    env.close()
    
    return input_size, output_size

def evaluate_fitness(structure, controller_weights, input_size, output_size, view=False):
    """Avalia a aptidão de uma combinação estrutura-controlador"""
    try:
        # Verificamos se a estrutura é válida
        if not is_connected(structure):
            return 0.0
            
        connectivity = get_full_connectivity(structure)
        env = gym.make(SCENARIO, max_episode_steps=STEPS, 
                      body=structure, connections=connectivity)
        
        # Criamos o controlador neural e configuramos seus pesos
        controller = NeuralController(input_size, output_size)
        set_weights(controller, controller_weights)
        
        # Preparamos o ambiente e o visualizador
        state = env.reset()[0]
        sim = env.sim
        
        if view:
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')
        
        total_reward = 0
        
        # Loop de simulação
        for t in range(STEPS):
            # Obtemos a ação do controlador neural
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = controller(state_tensor).detach().numpy().flatten()
            
            if view:
                viewer.render('screen')
                
            # Executamos a ação e obtemos a recompensa
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        if view:
            viewer.close()
        env.close()
        
        return total_reward
        
    except (ValueError, IndexError) as e:
        print(f"Error in evaluation: {e}")
        return 0.0

def evaluate_all_combinations(structure_population, controller_population, input_size, output_size):
    """Avalia todas as combinações estrutura-controlador"""
    num_structures = len(structure_population)
    num_controllers = len(controller_population)
    fitness_matrix = np.zeros((num_structures, num_controllers))
    
    for i, structure in enumerate(structure_population):
        for j, controller in enumerate(controller_population):
            fitness = evaluate_fitness(structure, controller, input_size, output_size)
            fitness_matrix[i, j] = fitness
            
    return fitness_matrix

def calculate_structure_fitness(fitness_matrix, i):
    """Calcula a fitness de uma estrutura com base nos K melhores controladores"""
    # Ordenamos os controladores pelo desempenho com esta estrutura
    controller_scores = fitness_matrix[i, :]
    top_k_scores = np.sort(controller_scores)[-NUM_PARTNERS:]
    return np.mean(top_k_scores)

def calculate_controller_fitness(fitness_matrix, j):
    """Calcula a fitness de um controlador com base nas K melhores estruturas"""
    # Ordenamos as estruturas pelo desempenho com este controlador
    structure_scores = fitness_matrix[:, j]
    top_k_scores = np.sort(structure_scores)[-NUM_PARTNERS:]
    return np.mean(top_k_scores)

# ----- FUNÇÃO PRINCIPAL DE COEVOLUÇÃO -----

def run_coevolution(num_generations=NUM_GENERATIONS):
    """Executa o algoritmo co-evolutivo"""
    # Inicialização
    print("Inicializando populações...")
    input_size, output_size = get_input_output_sizes()
    structure_population = initialize_structure_population()
    controller_population, cma_es, base_model = initialize_controller_population(input_size, output_size)
    
    # Variáveis para acompanhar o melhor encontrado
    best_structure = None
    best_controller = None
    best_fitness = -np.inf
    best_combination = (0, 0)
    
    # Histórico de métricas
    history = []
    
    # Tempo total
    start_time = time.time()
    
    # Salvar a geração inicial
    print("Salvando a geração inicial...")
    fitness_matrix = evaluate_all_combinations(structure_population, controller_population, input_size, output_size)
    parameters = {
        "scenario": SCENARIO,
        "mutation_rate": MUTATION_RATE,
        "crossover_rate": CROSSOVER_RATE,
        "tournament_size": TOURNAMENT_SIZE,
        "sigma_init": SIGMA_INIT,
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "seed": SEED
    }
    save_generation_data(
        generation=0,
        structures=structure_population,
        controllers=controller_population,
        fitness_matrix=fitness_matrix,
        best_combination=best_combination,
        best_fitness=best_fitness,
        scenario=SCENARIO,
        seed=SEED,
        parameters=parameters
    )
    
    # Loop principal
    for gen in range(1, num_generations + 1):  # Começa da geração 1
        generation_start = time.time()
        
        # 1. Avaliação de todas as combinações
        fitness_matrix = evaluate_all_combinations(structure_population, controller_population, input_size, output_size)
        
        # 2. Encontrar a melhor combinação desta geração
        best_i, best_j = np.unravel_index(np.argmax(fitness_matrix), fitness_matrix.shape)
        current_best_fitness = fitness_matrix[best_i, best_j]
        
        # 3. Atualizar o melhor global se necessário
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_structure = structure_population[best_i].copy()
            best_controller = controller_population[best_j].copy()
            best_combination = (best_i, best_j)
            print(f"Nova melhor combinação! Fitness: {best_fitness:.2f}")
        
        # 4. Calcular fitness para cada estrutura e controlador
        structure_fitness = np.array([calculate_structure_fitness(fitness_matrix, i) 
                                     for i in range(len(structure_population))])
        
        controller_fitness = np.array([calculate_controller_fitness(fitness_matrix, j) 
                                      for j in range(len(controller_population))])
        
        # 5. Evolução das populações
        structure_population = evolve_structures(structure_population, structure_fitness)
        controller_population = evolve_controllers(controller_population, cma_es, base_model, controller_fitness)
        
        # 6. Calcular tempo da geração
        generation_time = time.time() - generation_start
        
        # 7. Registrar histórico
        history.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(fitness_matrix),
            'gen_best_fitness': current_best_fitness,
            'time_per_gen': generation_time
        })
        
        # 8. Reportar progresso
        print(f"Geração {gen}/{num_generations}: Melhor Fitness = {current_best_fitness:.2f}, " + 
              f"Global = {best_fitness:.2f}, Tempo = {generation_time:.2f}s")
        
       
        save_generation_data(
                generation=gen,
                structures=structure_population,
                controllers=controller_population,
                fitness_matrix=fitness_matrix,
                best_combination=best_combination,
                best_fitness=best_fitness,
                scenario=SCENARIO,
                seed=SEED,
                parameters=parameters
        )
    
    # Calcular tempo total
    total_time = time.time() - start_time
    print(f"\nEvolução completa! Tempo total: {total_time:.2f}s")
    print(f"Melhor fitness global: {best_fitness:.2f}")
    
    # Salvar resultados finais
    save_history(history, f"coevolution_GA_CMAES_{SCENARIO}_seed{SEED}_history.csv")
    
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
        seed=SEED
    )
    
    return best_structure, best_controller, best_fitness

def visualize_best(structure, controller, input_size, output_size):
    """Visualiza o comportamento do robô especificado"""
    evaluate_fitness(structure, controller, input_size, output_size, view=True)

def create_gif(robot_structure, controller_weights, input_size, output_size, filename="best_robot.gif", 
               scenario=None, steps=500, duration=0.066):
    """
    Cria um GIF do robô em ação usando um controlador neural.
    
    Args:
        robot_structure: Estrutura do robô.
        controller_weights: Pesos do controlador neural.
        input_size: Tamanho do vetor de entrada do controlador.
        output_size: Tamanho do vetor de saída do controlador.
        filename: Nome do arquivo GIF a ser salvo.
        scenario: Nome do cenário do ambiente.
        steps: Número máximo de passos na simulação.
        duration: Duração de cada frame no GIF.
    """
    try:
        # Preparar o ambiente
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=steps, 
                       body=robot_structure, connections=connectivity)
        
        # Configurar o controlador neural
        controller_model = NeuralController(input_size, output_size)
        set_weights(controller_model, controller_weights)
        
        # Preparar simulação
        state = env.reset()[0]
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        
        # Preparar capturas de frames
        frames = []
        total_reward = 0
        
        # Loop de simulação
        for t in range(steps):
            # Obter ação do controlador
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = controller_model(state_tensor).detach().numpy().flatten()
            
            # Executar ação no ambiente
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Capturar frame
            frame = viewer.render('rgb_array')
            frames.append(frame)
            
            if terminated or truncated:
                break
        
        viewer.close()
        env.close()
        
        # Criar GIF
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
        print(f"GIF salvo em {filename}. Recompensa total: {total_reward}")
        
    except Exception as e:
        print(f"Erro ao criar GIF: {e}")

def run(num_runs=5):
    """Executa o algoritmo em múltiplos cenários e controladores com diferentes sementes."""
    scenarios = ['GapJumper-v0', 'CaveCrawler-v0']  # Cenários
    controllers = [NeuralController]  # Controladores a serem testados    
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
                
                # Executar o algoritmo
                start_time = time.time()
                best_structure, best_controller, best_fitness = run_coevolution()
                end_time = time.time()
                
                # Obter dimensões para visualização
                input_size, output_size = get_input_output_sizes()
                
                # Salvar GIF do melhor robô
                create_gif(
                    robot_structure=best_structure,
                    controller_weights=best_controller,
                    input_size=input_size,
                    output_size=output_size,
                    filename=f"{controller.__name__}_{scenario}_seed{RUN_SEED}.gif",
                    scenario=SCENARIO,
                    steps=500,
                    duration=0.066
                )
                
                # Registrar resultados
                final_results.append({
                    'scenario': scenario,
                    'controller': controller.__name__,
                    'seed': RUN_SEED,
                    'best_fitness': best_fitness,
                    'execution_time': round(end_time - start_time, 2)
                })
    
    # Salvar resultados finais em um DataFrame
    final_df = pd.DataFrame(final_results)
    final_df.to_excel('coevolution_multiple_scenarios_summary.xlsx', index=False)
    
    # Mostrar resumo
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
        # Configurar a seed
        print("\nConfigurando a seed...")
        RUN_SEED = SEED  # Use a seed global ou defina uma nova
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
        
        # Visualizar o melhor robô várias vezes
        print("\nVisualizando o melhor robô...")
        for _ in range(3):  # Visualizar 3 vezes
            visualize_best(best_structure, best_controller, input_size, output_size)
        
        '''
        # Simular o melhor robô 10 vezes
        print("\nSimulando o melhor robô várias vezes...")
        for i in range(10):
            utils.simulate_best_robot(best_structure, scenario=SCENARIO, steps=STEPS, controller=best_brain)
        '''
        # Criar GIF do melhor robô
        print("\nCriando GIF do melhor robô...")
        utils.create_gif(
            robot_structure=best_structure,
            controller_weights=best_controller,
            input_size=input_size,
            output_size=output_size,
            filename=f'{NeuralController.__name__}_{SCENARIO}_seed{RUN_SEED}.gif',
            scenario=SCENARIO,
            steps=STEPS,
            controller=best_brain
        )
        
        # Salvar pesos do controlador
        print("\nSalvando pesos do controlador...")
        torch.save(best_brain.state_dict(), f"best_controller_{SCENARIO}_seed{RUN_SEED}.pt")
        
        # Salvar estrutura do robô
        print("\nSalvando estrutura do robô...")
        np.save(f"best_structure_{SCENARIO}_seed{RUN_SEED}.npy", best_structure)
        
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
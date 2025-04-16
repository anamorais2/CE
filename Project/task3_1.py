import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
from joblib import Parallel, delayed
import json

# ---- PARAMETERS ----
NUM_GENERATIONS = 250  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid
STEPS = 500 # Number of simulation steps
SCENARIO = 'Walker-v0'
# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

CONTROLLER = alternating_gait # Controller to be used for simulation da para mudar

MUTATION_RATE = 0.5  # Probability of mutation -  ou seja há 20% de chance de a mutação ser aplicada. Usamos para evitar que o robô fique preso em um local.
# Mutation types: "Random Point", "Gaussian", "Swap"
CROSSOVER_RATE = 0.5  # Probability of crossover
POPULATION_SIZE = 50  # Number of robot structures per generation
SELECTION_TYPE = "Tournament" # Selection type (e.g., "Tournament", "Roulette")
TOURNAMENT_SIZE = 3  # Size of the tournament for selection

RUN_SEED = None  
ALGORITHM_NAME = "GA"  # pode ser "GA", "ES", etc.


#Abordagens não uteis no 3.1 - Abordagens que lidam com dominios continuos não são aplicáveis aqui - O CMA-ES, DE e PSO são algoritmos de otimização que funcionam melhor em espaços contínuos, enquanto o algoritmo genético é mais adequado para espaços discretos como o nosso.
# Representação de inteiros - vantagem - o individuo tinha o tamanho fixo, o que é bom para o nosso problema.
# fitness zero - penalizar robôs desconectados com um fitness muito baixo - o fitness é a recompensa total obtida pelo robô durante a simulação. Se o robô não se mover ou não conseguir coletar recompensas, o fitness será zero.
# O minino numero de runs - são 5 - Tirar conclusões especificas. Gráfico com a média mais o desvio padrão, podemos guardar as fitness scores de cada execução (geração) e depois calcular a média e o desvio padrão para cada geração. Isso nos dará uma ideia melhor da variabilidade dos resultados.
# Tabela - fitness / geração (numa run testar muitos geraões e para as outras runs, excluir as gerações que torna o gráfico mais constante) - Eficiencia- chegar ao resultado maior num menor espaço de tempo  (menor geraçao) - boas fitness acima de 4. Mais rápido com menos recursos
# Eficácia -  em 5 runs - perceber se é eficaz
# Tem de haver um equilibrio GA 50 pupulação x 100 gerações vs ES 5 x 100 gerações - o GA é mais eficiente, mas o ES é mais eficaz. O GA pode encontrar soluções boas mais rapidamente, enquanto o ES pode levar mais tempo, mas tende a encontrar soluções melhores em geral.



def save_generation_data(generation, population, fitness_scores, rewards, mutation_types, scenario, controller_name, seed, parameters):
    """
    Guarda os dados de uma geração inteira num CSV separado por seed.
    """
    folder = f"results_seed_{seed}"
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, f"generation_{generation}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Cabeçalho com metadados da run (apenas na primeira geração)
        if generation == 0:
            writer.writerow(["# ALGORITHM", ALGORITHM_NAME])
            for key, value in parameters.items():
                writer.writerow([f"# {key}", value])

        # Cabeçalho dos dados por indivíduo
        writer.writerow(["Index", "Fitness", "Reward", "FitnessNorm", "MutationType", "Connected", "Structure"]) #FitnessNorm baseada apenas na reward

        max_theoretical_reward = parameters.get("max_theoretical_reward", 1.0)

        for i, (individual, fitness, reward, mtype) in enumerate(zip(population, fitness_scores, rewards, mutation_types)):
            connected = is_connected(individual)
            structure_flat = individual.flatten().tolist()
            fitness_norm = reward / max_theoretical_reward if max_theoretical_reward else 0.0
            writer.writerow([i, fitness, reward, fitness_norm, mtype, connected, structure_flat])

        
def save_results_to_excel(controller, best_fitness, best_robot, mutation_type, mutation_rate, crossover_type, crossover_rate, scenario, population_size, num_generations, selection_type, tournament_size, execution_time, seed, filename='task3_1_Results_Complete.xlsx'):
    new_data = {
        'Scenario': [scenario],
        'Controller': [controller.__name__],
        'Population Size': [population_size],
        'Number of Generations': [num_generations],
        'Best Fitness': [best_fitness],
        'Best Robot Structure': [str(best_robot)],
        'Mutation Type': [mutation_type],
        'Mutation Rate': [mutation_rate],
        'Crossover Type': [crossover_type],
        'Crossover Rate': [crossover_rate],
        'Selection Type': [selection_type],
        'Tournament Size': [tournament_size if selection_type == "Tournament" else None],
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
    print(f"Resultados salvos em {filename}")
    
    

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


def select_parents(population, fitness_scores, selection_type="Tournament", tournament_size=3):
    if selection_type == "Tournament":
        selected = []
        for _ in range(2):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1]) 
            selected.append(winner[0])
        return selected[0], selected[1]
    
    elif selection_type == "Roulette":
        total_fitness = sum(fitness_scores)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        parent1 = random.choices(population, weights=probabilities, k=1)[0]
        parent2 = random.choices(population, weights=probabilities, k=1)[0]
        return parent1, parent2
    
    else:
        raise ValueError("Invalid selection type. Choose 'Tournament' or 'Roulette'.")



def evaluate_fitness(robot_structure, view=False):    
    try:
        if not is_connected(robot_structure):
            print("Robô desconectado!")
            return -100.0  # Penalizar robôs desconectados
  
        connectivity = get_full_connectivity(robot_structure)
        
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        obs = env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        
        t_reward = 0
        action_size = sim.get_dim_action_space('robot') 
        
        # Para retirar a posição e velocidade
        initial_pos = sim.object_pos_at_time(0, 'robot')
        positions = [initial_pos]
        max_distance = 0
        stability_penalty = 0
        energy_usage = 0
        backward_steps = 0
        voxel_count = np.sum(robot_structure > 0)  # Contar voxels não vazios

        for t in range(STEPS):  
            actuation = CONTROLLER(action_size, t) 
            
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            
            # Armazenar posição atual
            current_pos = sim.object_pos_at_time(t, 'robot')
            positions.append(current_pos)
            
            # Calcular distância percorrida (movimento horizontal usando a média das coordenadas X)
            distance_traveled = np.mean(current_pos[0]) - np.mean(initial_pos[0])  # Distância no eixo X
            max_distance = max(max_distance, distance_traveled)
            
            # Verificar movimento para trás
            if len(positions) > 1:
                if np.mean(current_pos[0]) < np.mean(positions[-2][0]):
                    backward_steps += 1
            
            # Verificar estabilidade (penalizar rotação excessiva)
            orientation = sim.object_orientation_at_time(t, 'robot')  # [0, 2pi]
            adjusted_orientation = (orientation + np.pi) % (2 * np.pi) - np.pi  # Ajustar para [-π, π]
            if abs(adjusted_orientation) > 0.5:  # Penalizar tombamento excessivo
                stability_penalty += 0.5
            
            # Penalizar uso de energia excessivo
            energy_usage += float(np.sum(np.abs(actuation)))
            
            if terminated or truncated:
                break

        viewer.close()
        env.close()
                
        # Calcular métricas de velocidade
        velocities = []
        for i in range(1, len(positions)):
            velocity = np.mean(positions[i][0]) - np.mean(positions[i-1][0])
            velocities.append(velocity)
            
        avg_velocity = float(np.mean(velocities)) if velocities else 0
        velocity_consistency = 1.0 / (float(np.std(velocities)) + 0.1) if velocities else 0
        
        # Definir valores máximos teóricos baseados no cenário
        if SCENARIO == 'Walker-v0':
            MAX_THEORETICAL_DISTANCE = STEPS * 0.1  
            MAX_THEORETICAL_VELOCITY = 0.5
            MAX_THEORETICAL_REWARD = STEPS * 0.1
        elif SCENARIO == 'BridgeWalker-v0':
            MAX_THEORETICAL_DISTANCE = STEPS * 0.08 
            MAX_THEORETICAL_VELOCITY = 0.4
            MAX_THEORETICAL_REWARD = STEPS * 0.08
        else:
            MAX_THEORETICAL_DISTANCE = STEPS * 0.1
            MAX_THEORETICAL_VELOCITY = 0.5
            MAX_THEORETICAL_REWARD = STEPS * 0.1
            
        MAX_STABILITY_PENALTY = STEPS * 0.5  # Máximo de penalidade por instabilidade
        MAX_ENERGY_USAGE = STEPS * action_size * 2  # Máximo teórico de uso de energia
        
        # Normalizar métricas com valores teóricos
        norm_distance = min(max_distance / MAX_THEORETICAL_DISTANCE, 1.0)
        norm_velocity = min(avg_velocity / MAX_THEORETICAL_VELOCITY, 1.0)
        norm_reward = min(t_reward / MAX_THEORETICAL_REWARD, 1.0)
        norm_stability = min(stability_penalty / MAX_STABILITY_PENALTY, 1.0)
        norm_energy = min(energy_usage / MAX_ENERGY_USAGE, 1.0)
        
        # Penalidades
        backward_penalty = backward_steps * 0.2
        complexity_penalty = voxel_count * 0.02  # Penalizar robôs muito complexos
        
        if SCENARIO == 'Walker-v0':
            fitness = (
                norm_distance * 2.0 +
                norm_velocity * 1.5 +
                velocity_consistency * 1.0 +
                norm_reward * 0.5 -
                norm_stability * 2.0 -
                complexity_penalty * 2.0 -
                backward_penalty * 1.5 -
                norm_energy * 0.5
            )
        elif SCENARIO == 'BridgeWalker-v0':
            fitness = (
                norm_distance * 0.8 +
                norm_velocity * 1.5 +
                velocity_consistency * 1.0 +
                norm_reward -
                norm_stability * 2 -
                complexity_penalty -
                backward_penalty -
                norm_energy * 0.3
            )
        else:
            fitness = (
                norm_distance +
                norm_velocity * 1.5 +
                velocity_consistency * 0.5 +
                norm_reward -
                norm_stability -
                complexity_penalty -
                backward_penalty -
                norm_energy * 0.2
            )
       
     

        return fitness, t_reward
        
    except (ValueError, IndexError, TypeError) as e:
        print(f"Erro na função de fitness: {e}")
        return -100.0
    

    
    
def mutate_diverse(robot_structure, generation, max_attempts=10):
    
    for attempt in range(max_attempts):
        mutated = copy.deepcopy(robot_structure)
        
        # Escolha baseada na fase da evolução
        phase = generation / NUM_GENERATIONS
        
        if phase < 0.3:  # Fase inicial: exploração ampla
            mutation_probs = {"point": 0.1, "block": 0.5, "add_remove": 0.2, "add_actuator": 0.2}
        elif phase < 0.7: # Fase média: equilíbrio
            mutation_probs = {"point": 0.3, "block": 0.3, "add_remove": 0.2, "add_actuator": 0.2}
        else: # Fase final: refinamento
            mutation_probs = {"point": 0.6, "block": 0.1, "add_remove": 0.2, "add_actuator": 0.1}
        
        mutation_type = random.choices(
            list(mutation_probs.keys()), 
            weights=list(mutation_probs.values())
        )[0]
        
        if mutation_type == "point":
            for i in range(5):
                for j in range(5):
                    if random.random() < MUTATION_RATE / 5: # Taxa reduzida para equilibrar
                        mutated[i, j] = random.choice(VOXEL_TYPES)
        
        elif mutation_type == "block":
            # Mutação de bloco (altera uma região 2x2)
            if random.random() < MUTATION_RATE:
                start_i = random.randint(0, 3)
                start_j = random.randint(0, 3)
                block_type = random.choice(VOXEL_TYPES)
                mutated[start_i:start_i+2, start_j:start_j+2] = block_type
        
        elif mutation_type == "add_remove":
            # Adicionar material onde está vazio ou remover onde tem material
            i, j = random.randint(0, 4), random.randint(0, 4)
            if mutated[i, j] == 0:
                mutated[i, j] = random.choice([1, 2, 3, 4])
            else:
                if random.random() < 0.5: # 50% de chance de remover, a menos que seja o único atuador
                     # Verificar se é um atuador e se é o único
                    if mutated[i, j] in [3, 4] and np.sum(np.isin(mutated, [3, 4])) <= 1:
                        pass
                    else:
                        mutated[i, j] = 0 # Não remover o único atuador
        
        elif mutation_type == "add_actuator":
            # Adicionar um novo atuador ou substituir um voxel existente por um atuador
            i, j = random.randint(0, 4), random.randint(0, 4)
            mutated[i, j] = random.choice([3, 4])
        
        # Garantir pelo menos um atuador
        if not np.any(np.isin(mutated, [3, 4])):
            i, j = random.randint(0, 4), random.randint(0, 4)
            mutated[i, j] = random.choice([3, 4])

        if is_connected(mutated):
            return mutated, mutation_type

    # Se falhar em todas as tentativas, retorna estrutura original com tipo de mutação como "none"
    return copy.deepcopy(robot_structure), "none"

    
 
# Mutação de um ponto aleatório
def mutate(robot_structure):
    mutated = copy.deepcopy(robot_structure)
    if random.random() < MUTATION_RATE:
        x, y = random.randint(0, 4), random.randint(0, 4)
        mutated[x, y] = random.choice(VOXEL_TYPES)
    return mutated

''' 
# Crossover de um ponto aleatório 
def crossover(robot_structure1, robot_structure2):
    if random.random() < CROSSOVER_RATE:
        x, y = random.randint(0, 4), random.randint(0, 4)
        child = copy.deepcopy(robot_structure1)
        child[x:, y:] = robot_structure2[x:, y:]
        return child
    return robot_structure1
'''
# Crossover uniforme

def crossover(robot_structure1, robot_structure2, max_attempts=10):
    for attempt in range(max_attempts):
        if random.random() < CROSSOVER_RATE:
            # Criar uma máscara aleatória para decidir de qual pai cada voxel será herdado
            mask = np.random.rand(*robot_structure1.shape) < 0.5  # Cada posição da máscara indica se o voxel será herdado do robot_structure1 (False) ou do robot_structure2 (True).
            
            # Para cada posição onde a máscara é True, o valor do robot_structure2 é copiado para o filho.
            # Para cada posição onde a máscara é False, o valor do robot_structure1 é mantido no filho.
            child = copy.deepcopy(robot_structure1)
            child[mask] = robot_structure2[mask]

            # Verificar se o robô gerado está conectado
            if is_connected(child):
                return child

    # Se todas as tentativas falharem, retornar o parent 1
    return copy.deepcopy(robot_structure1)


def save_history(history, filename):
    df = pd.DataFrame(history)
    
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df
    
    combined_df.to_csv(filename, index=False)
    print(f"Histórico salvo em {filename}")
    
    
def plot_evolution_metrics(history, filename):
    """Cria gráficos de evolução das métricas"""
    plt.figure(figsize=(12, 8))
    
    # Gráfico de fitness
    plt.subplot(2, 2, 1)
    plt.plot(history['generation'], history['best_fitness'], 'b-', label='Melhor da geração')
    plt.plot(history['generation'], history['max_fitness_ever'], 'r-', label='Melhor até agora')
    plt.fill_between(history['generation'], 
                    [a - b for a, b in zip(history['avg_fitness'], history['std_fitness'])], 
                    [a + b for a, b in zip(history['avg_fitness'], history['std_fitness'])], 
                    alpha=0.2, color='gray')
    plt.plot(history['generation'], history['avg_fitness'], 'g-', label='Média')
    plt.title('Evolução da Fitness')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de tempo por geração
    plt.subplot(2, 2, 2)
    plt.plot(history['generation'], history['time_per_gen'], 'k-')
    plt.title('Tempo por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Tempo (s)')
    plt.grid(True)
    
    
    # Salvar figura
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"Gráfico salvo em {filename}")
    
def evaluate_individual(individual):
    """Avalia a fitness de um indivíduo (robô) em paralelo."""
    from evogym.envs import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
    fitness, reward = evaluate_fitness(individual)
    return individual, fitness, reward


# ---- EVOLUTIONARY ALGORITHM ----
def evolve():
    global RUN_SEED
    population = create_population()
    best_robot = None
    best_fitness = -float('inf')
    
    history = {
        'generation': [],
        'best_fitness': [],
        'avg_fitness': [],
        'std_fitness': [],
        'max_fitness_ever': [],
        'time_per_gen': []
    }
    
    all_parameters = {
        "scenario": SCENARIO,
        "controller": CONTROLLER.__name__,
        "mutation_rate": MUTATION_RATE,
        "crossover_rate": CROSSOVER_RATE,
        "selection_type": SELECTION_TYPE,
        "tournament_size": TOURNAMENT_SIZE,
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "seed": RUN_SEED,
        "max_theoretical_reward": STEPS * 0.1 if SCENARIO == 'Walker-v0' else STEPS * 0.08
    }
    
   
    
    
    for generation in range(NUM_GENERATIONS):
        
        start_time = time.time()
        
        fitness_scores = []
        rewards = []
        mutation_types = []  
        
        individuals_data = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind) for ind in population)
        for ind, fit, rew in individuals_data:
            fitness_scores.append(fit)
            rewards.append(rew)
            
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        rewards = [rewards[i] for i in sorted_indices]
        
        # Guardar geração 0 antes de qualquer mutação
        if generation == 0:
            save_generation_data(
                generation=generation,
                population=population,
                fitness_scores=fitness_scores,
                rewards=rewards,
                mutation_types=["Inicial"] * len(population),
                scenario=SCENARIO,
                controller_name=CONTROLLER.__name__,
                seed=RUN_SEED,
                parameters=all_parameters
            )

        #Elitismo 
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            best_robot = copy.deepcopy(population[0])

        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        history['generation'].append(generation)
        history['best_fitness'].append(fitness_scores[0])
        history['avg_fitness'].append(avg_fitness)
        history['std_fitness'].append(std_fitness)
        history['max_fitness_ever'].append(best_fitness)
        history['time_per_gen'].append(time.time() - start_time)
        
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}, Best Fitness: {best_fitness}")
        
        
        new_population = []
    
        new_population.append(population[0])  # Manter o melhor robô da geração anterior
        new_population.append(population[1])  # Manter o segundo melhor robô da geração anterior
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores, selection_type=SELECTION_TYPE)
            child = crossover(parent1, parent2)
            child, mutate_type = mutate_diverse(child, generation)
            new_population.append(child)
            mutation_types.append(mutate_type)
            
        if generation > 0:
            save_generation_data(
                generation=generation,
                population=new_population,
                fitness_scores=fitness_scores,
                rewards=rewards,
                mutation_types=mutation_types,
                scenario=SCENARIO,
                controller_name=CONTROLLER.__name__,
                seed=RUN_SEED,
                parameters=all_parameters
            )
        
        population = new_population
        
       
        
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
        
    
    save_history(history, f"{CONTROLLER.__name__}_{SCENARIO}_history_seed{RUN_SEED}.csv")
    #plot_evolution_metrics(history, f"{CONTROLLER.__name__}_{SCENARIO}_evolution_seed{RUN_SEED}.png")
    
    return best_robot, best_fitness



# Caminho para salvar o progresso
progress_file = "parameter_test_progress.json"

# Função para salvar progresso
def save_progress(results):
    with open(progress_file, "w") as file:
        json.dump(results, file)
    print(f"Progresso salvo em {progress_file}")

# Função para carregar progresso
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            return json.load(file)
    return []

def run_parameter_test():
    global RUN_SEED, NUM_GENERATIONS  
    base_seed = 123  # Seed fixa para consistência
    
    # Parâmetros a testar
    controllers = [alternating_gait, sinusoidal_wave, hopping_motion]
    mutation_rates = [0.1, 0.5, 0.9]
    crossover_rates = [0.3, 0.5, 0.7]
    tournament_sizes = [2, 3, 5]
    
    # Carregar progresso salvo
    results = load_progress()
    tested_combinations = {(res['controller'], res['mutation_rate'], res['crossover_rate'], res['tournament_size']) for res in results}
    
    # Para cada combinação
    for controller in controllers:
        for mut_rate in mutation_rates:
            for cross_rate in crossover_rates:
                for tourn_size in tournament_sizes:
                    # Verificar se a combinação já foi testada
                    if (controller.__name__, mut_rate, cross_rate, tourn_size) in tested_combinations:
                        continue
                    
                    print(f"\nTestando: Controller={controller.__name__}, MutRate={mut_rate}, CrossRate={cross_rate}, TournSize={tourn_size}")
                    
                    # Configurar parâmetros globais
                    global CONTROLLER, MUTATION_RATE, CROSSOVER_RATE, TOURNAMENT_SIZE
                    CONTROLLER = controller
                    MUTATION_RATE = mut_rate
                    CROSSOVER_RATE = cross_rate
                    TOURNAMENT_SIZE = tourn_size
                    
                    # Executar 3 testes rápidos (menos gerações para teste inicial)
                    test_generations = 10  # Menos gerações para testes rápidos
                    temp_num_gen = NUM_GENERATIONS  # Salvar o valor original de NUM_GENERATIONS
                    NUM_GENERATIONS = test_generations  # Alterar temporariamente
                    
                    test_results = []
                    for test in range(3):
                        RUN_SEED = base_seed + test
                        random.seed(RUN_SEED)
                        np.random.seed(RUN_SEED)
                        print(f"Teste {test+1}/3 com seed {RUN_SEED}")
                        best_robot, best_fitness = evolve()
                        test_results.append(best_fitness)

                    # Restaurar número de gerações
                    NUM_GENERATIONS = temp_num_gen
                    
                    # Salvar resultados
                    avg_fitness = np.mean(test_results)
                    results.append({
                        'controller': controller.__name__,
                        'mutation_rate': mut_rate,
                        'crossover_rate': cross_rate,
                        'tournament_size': tourn_size,
                        'avg_fitness': avg_fitness,
                        'tests': test_results
                    })
                    
                    # Salvar progresso parcial
                    save_progress(results)
                    
                    print(f"Resultado médio: {avg_fitness:.2f}")
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results)
    results_df.to_excel('parameter_testing_results.xlsx', index=False)
    
    # Identificar melhores parâmetros
    best_config = results_df.loc[results_df['avg_fitness'].idxmax()]
    print("\nMelhor configuração encontrada:")
    print(f"Controller: {best_config['controller']}")
    print(f"Mutation Rate: {best_config['mutation_rate']}")
    print(f"Crossover Rate: {best_config['crossover_rate']}")
    print(f"Tournament Size: {best_config['tournament_size']}")
    print(f"Average Fitness: {best_config['avg_fitness']:.2f}")
    
    return results_df

def main_teste():
    global RUN_SEED
    
    mode = input("Selecione o modo (1=Run único, 2=Teste de parâmetros, 3=Execução completa): ")
    
    if mode == '1':
        global CONTROLLER, SCENARIO, POPULATION_SIZE, NUM_GENERATIONS, SELECTION_TYPE, TOURNAMENT_SIZE
        RUN_SEED = 42
        random.seed(RUN_SEED)
        np.random.seed(RUN_SEED)
        start = time.time()
        best_robot, best_fitness = evolve()
        end = time.time()
        print(f"Melhor robô encontrado com fitness {best_fitness}:")
        print(best_robot)
        
        save_results_to_excel(
            controller=CONTROLLER,
            best_fitness=best_fitness,
            best_robot=best_robot,
            mutation_type="Diverse",
            mutation_rate=MUTATION_RATE,
            crossover_type="Uniform",
            crossover_rate=CROSSOVER_RATE,
            scenario=SCENARIO,
            population_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            selection_type=SELECTION_TYPE,
            tournament_size=TOURNAMENT_SIZE,
            execution_time=round(end - start, 2),
            seed=RUN_SEED
        )

    
        i = 0
        while i < 10:
            utils.simulate_best_robot(best_robot, scenario=SCENARIO, steps=STEPS)
            i += 1
            
        utils.create_gif(best_robot, filename=f'{CONTROLLER.__name__}_{SCENARIO}_seed{RUN_SEED}.gif',
                         scenario=SCENARIO, steps=STEPS, controller=CONTROLLER)
        
    elif mode == '2':
        results_df = run_parameter_test()
        print("Testes de parâmetros concluídos. Resultados salvos em 'parameter_testing_results.xlsx'")
        
    elif mode == '3':
        scenarios = ['Walker-v0', 'BridgeWalker-v0']
        controllers = [alternating_gait, sinusoidal_wave, hopping_motion]
        
        final_results = []
        
        for scenario in scenarios:
            for controller in controllers:
                SCENARIO = scenario
                CONTROLLER = controller
                
                print(f"\n\nExecutando {controller.__name__} em {scenario}")
                
                # Executar 5 testes
                for seed in range(5):
                    RUN_SEED = 42 + seed
                    random.seed(RUN_SEED)
                    np.random.seed(RUN_SEED)
                    
                    print(f"\nExecução {seed+1}/5 com seed {RUN_SEED}")
                    start = time.time()
                    best_robot, best_fitness = evolve()
                    end = time.time()
          
                    save_results_to_excel(
                        controller=controller,
                        best_fitness=best_fitness,
                        best_robot=best_robot,
                        mutation_type="Diverse",
                        mutation_rate=MUTATION_RATE,
                        crossover_type="Uniform",
                        crossover_rate=CROSSOVER_RATE,
                        scenario=scenario,
                        population_size=POPULATION_SIZE,
                        num_generations=NUM_GENERATIONS,
                        selection_type=SELECTION_TYPE,
                        tournament_size=TOURNAMENT_SIZE,
                        execution_time=round(end - start, 2),
                        seed=RUN_SEED,
                        filename=f'final_results_summary.xlsx'
                    )
                    
                    if seed == 0 or best_fitness > final_results[-1]['best_fitness'] if final_results else True:
                        utils.create_gif(
                            best_robot, 
                            filename=f'{controller.__name__}_{scenario}_seed{seed}.gif', 
                            scenario=scenario, 
                            steps=STEPS, 
                            controller=controller
                        )
                    
                    final_results.append({
                        'scenario': scenario,
                        'controller': controller.__name__,
                        'seed': seed,
                        'best_fitness': best_fitness
                    })
        
        # Análise final
        final_df = pd.DataFrame(final_results)
        final_df.to_excel('experiment_summary.xlsx', index=False)
        
        # Mostrar resumo
        print("\nResumo do Experimento:")
        for scenario in scenarios:
            for controller_name in [c.__name__ for c in controllers]:
                subset = final_df[(final_df['scenario'] == scenario) & (final_df['controller'] == controller_name)]
                avg = subset['best_fitness'].mean()
                std = subset['best_fitness'].std()
                print(f"{scenario} + {controller_name}: Média={avg:.2f}, Desvio={std:.2f}")
    
    else:
        print("Modo inválido selecionado.")



if __name__ == '__main__':
    main_teste()
  
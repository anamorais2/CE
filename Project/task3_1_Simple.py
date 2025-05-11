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
ALGORITHM_NAME = "Simple_GA"  # Identificação do algoritmo


#Abordagens não uteis no 3.1 - Abordagens que lidam com dominios continuos não são aplicáveis aqui - O CMA-ES, DE e PSO são algoritmos de otimização que funcionam melhor em espaços contínuos, enquanto o algoritmo genético é mais adequado para espaços discretos como o nosso.
# Representação de inteiros - vantagem - o individuo tinha o tamanho fixo, o que é bom para o nosso problema.
# fitness zero - penalizar robôs desconectados com um fitness muito baixo - o fitness é a recompensa total obtida pelo robô durante a simulação. Se o robô não se mover ou não conseguir coletar recompensas, o fitness será zero.
# O minino numero de runs - são 5 - Tirar conclusões especificas. Gráfico com a média mais o desvio padrão, podemos guardar as fitness scores de cada execução (geração) e depois calcular a média e o desvio padrão para cada geração. Isso nos dará uma ideia melhor da variabilidade dos resultados.
# Tabela - fitness / geração (numa run testar muitos geraões e para as outras runs, excluir as gerações que torna o gráfico mais constante) - Eficiencia- chegar ao resultado maior num menor espaço de tempo  (menor geraçao) - boas fitness acima de 4. Mais rápido com menos recursos
# Eficácia -  em 5 runs - perceber se é eficaz
# Tem de haver um equilibrio GA 50 pupulação x 100 gerações vs ES 5 x 100 gerações - o GA é mais eficiente, mas o ES é mais eficaz. O GA pode encontrar soluções boas mais rapidamente, enquanto o ES pode levar mais tempo, mas tende a encontrar soluções melhores em geral.



def save_generation_data(generation, population, fitness_scores, mutation_types, scenario, controller_name, seed, parameters):
  
    folder = f"results_seed_{seed}/{controller_name}_{scenario}_Simple"
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
        writer.writerow(["Index", "Fitness", "Mutation_type","Connected", "Structure"]) 

        for i, (individual, fitness, mtype) in enumerate(zip(population, fitness_scores, mutation_types)):
            connected = is_connected(individual)
            structure_flat = individual.flatten().tolist()
            writer.writerow([i, fitness, mtype, connected, structure_flat])

        
def save_results_to_excel(controller, best_fitness, best_robot, mutation_type, mutation_rate, crossover_type, crossover_rate, scenario, population_size, num_generations, selection_type, tournament_size, execution_time, seed, filename):
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


# ---- SELECTION ----
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


# ---- FITNESS EVALUATION ----
def evaluate_fitness(robot_structure, view=False):    
    try:
        if not is_connected(robot_structure):
            print("Robô desconectado!")
            return -100.0
  
        connectivity = get_full_connectivity(robot_structure)
        
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=robot_structure, connections=connectivity)
        obs = env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        
        t_reward = 0
        action_size = sim.get_dim_action_space('robot') 
        


        for t in range(STEPS):  
            actuation = CONTROLLER(action_size, t) 
            
            if view:
                viewer.render('screen') 
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            

            
            if terminated or truncated:
                break

        viewer.close()
        env.close()
                

        return t_reward
        
    except (ValueError, IndexError, TypeError) as e:
        print(f"Erro na função de fitness: {e}")
        return -100.0
    

    
    

 # ---- MUTATION ----
def mutate_simple(robot_structure, max_attempts=10):
    
    for attempt in range(max_attempts):
        mutated = copy.deepcopy(robot_structure)
        
        # Aplicar mutação em alguns pontos aleatórios com base na taxa de mutação
        for i in range(robot_structure.shape[0]):
            for j in range(robot_structure.shape[1]):
                if random.random() < MUTATION_RATE:
                    mutated[i, j] = random.choice(VOXEL_TYPES)
        
        # Garantir que o robô tenha pelo menos um atuador
        if not np.any(np.isin(mutated, [3, 4])):
            i, j = random.randint(0, mutated.shape[0]-1), random.randint(0, mutated.shape[1]-1)
            mutated[i, j] = random.choice([3, 4])
        
        # Verificar se o robô gerado está conectado
        if is_connected(mutated):
            return mutated

    return copy.deepcopy(robot_structure)


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
    
    
def evaluate_individual(individual):
    """Avalia a fitness de um indivíduo (robô) em paralelo."""
    from evogym.envs import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
    fitness = evaluate_fitness(individual)
    return individual, fitness 


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
    

    total_start_time = time.time()
    
    
    for generation in range(NUM_GENERATIONS):
        
        start_time = time.time()
        
        fitness_scores = []
        mutation_types = []  
        
        individuals_data = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind) for ind in population)
        for ind, fit in individuals_data:
            fitness_scores.append(fit)
            
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        
        # Guardar geração 0 antes de qualquer mutação
        if generation == 0:
            save_generation_data(
                generation=generation,
                population=population,
                fitness_scores=fitness_scores,
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
            child= mutate_simple(child)
            new_population.append(child)
            
        if generation > 0:
            save_generation_data(
                generation=generation,
                population=new_population,
                fitness_scores=fitness_scores,
                mutation_types= ["Simple"] * len(population),
                scenario=SCENARIO,
                controller_name=CONTROLLER.__name__,
                seed=RUN_SEED,
                parameters=all_parameters
            )
        
        population = new_population
        
       
        
    total_time = time.time() - total_start_time
    print(f"Total time: {total_time:.2f} seconds")
        
    
    save_history(history, f"{CONTROLLER.__name__}_{SCENARIO}__{ALGORITHM_NAME}_history_seed{RUN_SEED}.csv")

    
    return best_robot, best_fitness




progress_file = "parameter_test_progress.json"


def save_progress(results):
    with open(progress_file, "w") as file:
        json.dump(results, file)
    print(f"Progresso salvo em {progress_file}")


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
    

    results_df = pd.DataFrame(results)
    results_df.to_excel('parameter_testing_results.xlsx', index=False)

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
        scenarios = ['BridgeWalker-v0']
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
                        mutation_type="Simple",
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
                    
                    utils.create_gif(
                            best_robot, 
                            filename=f'{controller.__name__}_{scenario}_{ALGORITHM_NAME}_seed{seed}.gif', 
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
   
        final_df = pd.DataFrame(final_results)
        final_df.to_excel('experiment_summary_GA_SIMPLE.xlsx', index=False)
        
  
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
  
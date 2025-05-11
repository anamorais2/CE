import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Função para carregar resultados de todas as execuções
def load_all_results(base_folder, algorithm):
    """
    Carrega os resultados de todas as execuções para ambos os cenários.
    Para cada geração, coleta o melhor reward.
    No final, imprime a média e o desvio padrão dos melhores rewards por geração para cada cenário.
    """
    all_results = []
    best_rewards_by_scenario = {
        'DownStepper-v0': [],
        'ObstacleTraverser-v0': []
    }

    # Para cada seed
    for seed in [51, 52, 53, 54, 55]:
        seed_folder = f"{base_folder}{seed}"

        # Para cada cenário
        for scenario in ['DownStepper-v0', 'ObstacleTraverser-v0']:
            scenario_folder = os.path.join(seed_folder, f"NeuralController_{scenario}_{algorithm}")

            if not os.path.exists(scenario_folder):
                print(f"Pasta não encontrada: {scenario_folder}")
                continue

            gen_files = glob.glob(os.path.join(scenario_folder, "generation_*.csv"))
            if not gen_files:
                print(f"Nenhum arquivo de geração encontrado em: {scenario_folder}")
                continue

            best_rewards_by_gen = []

            for gen_file in gen_files:
                try:
                    gen_data = pd.read_csv(gen_file, comment="#")
                    best_in_gen = max(gen_data['Reward'])
                    best_rewards_by_gen.append(best_in_gen)
                except Exception as e:
                    print(f"Erro ao carregar {gen_file}: {e}")

            if best_rewards_by_gen:
                best_reward_overall = max(best_rewards_by_gen)
                all_results.append({
                    'Seed': seed,
                    'Scenario': scenario,
                    'Best_reward': best_reward_overall
                })
                best_rewards_by_scenario[scenario].extend(best_rewards_by_gen)

    # Print das estatísticas por cenário
    for scenario in ['DownStepper-v0', 'ObstacleTraverser-v0']:
        rewards = best_rewards_by_scenario[scenario]
        if rewards:
            avg = np.mean(rewards)
            std = np.std(rewards)
            print(f"{algorithm} - {scenario}")
            print(f"→ Média dos melhores rewards de cada geração: {avg:.4f}")
            print(f"→ Desvio padrão: {std:.4f}\n")

    return pd.DataFrame(all_results)

# Função para análise descritiva
def descriptive_analysis(results_df):
    """
    Realiza análise descritiva dos resultados.
    """
    print("=== Análise Descritiva ===")
    
    # Estatísticas por cenário
    print("\nEstatísticas por cenário:")
    stats_by_scenario = results_df.groupby('Scenario')['Best_reward'].describe()
    print(stats_by_scenario)
    
    # Cria boxplots para visualizar a distribuição dos rewards
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Scenario', y='Best_reward', data=results_df)
    plt.title('Distribuição de reward por Cenário')
    plt.tight_layout()
    plt.savefig('boxplot_reward_by_scenario.png')
    plt.close()
    
    return stats_by_scenario

# Função para verificar normalidade
def check_normality(results_df):
    """
    Verifica a normalidade dos dados usando o teste de Shapiro-Wilk.
    """
    print("\n=== Teste de Normalidade (Shapiro-Wilk) ===")
    
    is_normal = {}
    
    for scenario in results_df['Scenario'].unique():
        data = results_df[results_df['Scenario'] == scenario]['Best_reward']
        stat, p_value = stats.shapiro(data)
        
        is_normal[scenario] = p_value > 0.05
        
        print(f"Cenário: {scenario}")
        print(f"Estatística do teste: {stat:.4f}")
        print(f"p-value: {p_value}")
        print(f"Conclusão: {'Distribuição normal' if is_normal[scenario] else 'Distribuição não normal'}\n")
    
    return is_normal

# Função de comparação entre cenários
def compare_scenarios(results_df, is_normal):
    """
    Compara os cenários usando teste t ou teste de Mann-Whitney U dependendo da normalidade.
    """
    scenarios = results_df['Scenario'].unique()
    
    if len(scenarios) <= 1:
        print("Não há cenários suficientes para comparação.")
        return
    
    # Obtém os dados para cada cenário
    data1 = results_df[results_df['Scenario'] == scenarios[0]]['Best_reward']
    data2 = results_df[results_df['Scenario'] == scenarios[1]]['Best_reward']
    
    # Verifica se ambos são normais
    if is_normal.get(scenarios[0], False) and is_normal.get(scenarios[1], False):
        # Teste t para amostras independentes
        print("\n=== Teste t para Amostras Independentes ===")
        stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Usando Welch's t-test
        
        print(f"Comparando {scenarios[0]} vs {scenarios[1]}")
        print(f"Estatística t: {stat:.4f}")
        print(f"p-value: {p_value}")
        print(f"Conclusão: {'Diferença significativa' if p_value < 0.05 else 'Não há diferença significativa'}")
    else:
        # Teste de Mann-Whitney U (não paramétrico)
        print("\n=== Teste de Mann-Whitney U ===")
        stat, p_value = stats.mannwhitneyu(data1, data2)
        
        print(f"Comparando {scenarios[0]} vs {scenarios[1]}")
        print(f"Estatística U: {stat:.4f}")
        print(f"p-value: {p_value}")
        print(f"Conclusão: {'Diferença significativa' if p_value < 0.05 else 'Não há diferença significativa'}")

# Função para comparar as recompensas entre seeds
def compare_seeds(results_df, is_normal, algorithm):
    """
    Compara as recompensas entre seeds para cada cenário usando ANOVA ou Kruskal-Wallis.
    """
    for scenario in results_df['Scenario'].unique():
        print(f"\n--- Comparando seeds para o cenário: {scenario} ---")

        seed_groups = []
        seed_labels = []

        for seed in results_df['Seed'].unique():
            scenario_folder = os.path.join(f"results_seed_{seed}", f"NeuralController_{scenario}_{algorithm}")
            gen_path = os.path.join(scenario_folder, "generation_99.csv")

            if os.path.exists(gen_path):
                try:
                    rewards = extract_rewards(gen_path, is_normal)
                    # Achata os arrays para garantir que temos uma lista simples de números
                    flat_rewards = np.concatenate(rewards)
                    seed_groups.append(flat_rewards)
                    seed_labels.append(f"Seed {seed}")
                except Exception as e:
                    print(f"Erro ao processar seed {seed} para cenário {scenario}: {e}")
            else:
                print(f"Arquivo não encontrado: {gen_path}")

        # Verificação de número mínimo de grupos
        if len(seed_groups) < 2:
            print("Não há grupos suficientes para comparação (mínimo 2).")
            continue

        if is_normal.get(scenario, False):
            print("Usando ANOVA")
            f_stat, p_value = stats.f_oneway(*seed_groups)
            print(f"Estatística F: {f_stat:.4f} | p-value: {p_value}")
        else:
            print("Usando Kruskal-Wallis")
            h_stat, p_value = stats.kruskal(*seed_groups)
            print(f"Estatística H: {h_stat:.4f} | p-value: {p_value}")

        print(f"Conclusão: {'Diferença significativa entre seeds' if p_value < 0.05 else 'Não há diferença significativa entre seeds'}")



# Função para extrair as recompensas de todas as gerações
def extract_rewards(file_path, is_normal):
    """
    Extrai os valores de recompensa de todas as gerações do arquivo CSV.
    """
    df = pd.read_csv(file_path, comment="#")
    all_rewards = []

    for index, row in df.iterrows():
        reward_val = row['Reward']

        try:
            if isinstance(reward_val, str):
                # Se a recompensa for uma string, tenta converter para um array
                cleaned = reward_val.strip('[]').replace('\n', '').strip()
                if cleaned:
                    reward_array = np.fromstring(cleaned, sep=',')
                else:
                    reward_array = np.array([])
            elif isinstance(reward_val, (int, float)):
                reward_array = np.array([reward_val])
            else:
                raise TypeError(f"Tipo inesperado em Reward: {type(reward_val)}")

            all_rewards.append(reward_array)

        except Exception as e:
            print(f"[Linha {index}] Erro ao processar Reward '{reward_val}': {e}")
            continue

    return all_rewards



# Função para a análise da progressão ao longo das gerações
def analyze_progression(base_folder, algorithm):
    """
    Analisa a progressão do reward ao longo das gerações.
    """
    print("\n=== Análise da Progressão do reward ===")
    
    # Para cada cenário
    for scenario in ['DownStepper-v0', 'ObstacleTraverser-v0']:
        plt.figure(figsize=(12, 8))
        
        # Para cada seed
        for seed in [51, 52, 53, 54, 55]:
            progression_data = []
            seed_folder = f"{base_folder}{seed}" 
            scenario_folder = os.path.join(seed_folder, f"NeuralController_{scenario}_{algorithm}") 
            
            # Verifica se a pasta existe
            if not os.path.exists(scenario_folder):
                continue
                
            # Encontra todos os arquivos da geração
            gen_files = glob.glob(os.path.join(scenario_folder, "generation_*.csv")) 
            
            for gen_file in sorted(gen_files, key=lambda x: int(x.split("_")[-1].split(".")[0])):
                gen_num = int(gen_file.split("_")[-1].split(".")[0])
                
                try:
                    gen_data = pd.read_csv(gen_file, comment="#")
                    # Pega na melhor reward da geração 
                    best_reward = max(gen_data['Reward'])
                    progression_data.append((gen_num, best_reward))
                except Exception as e:
                    print(f"Erro ao carregar {gen_file}: {e}")
            
            if progression_data:
                gens, rewardes = zip(*progression_data)
                plt.plot(gens, rewardes, label=f"Seed {seed}")
        
        plt.title(f'Progressão do reward ao Longo das Gerações - {scenario}')
        plt.xlabel('Geração')
        plt.ylabel('Melhor reward')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'progression_{scenario.replace("-", "_")}_{algorithm}.png')
        plt.close()
        
# Função para comparar os algoritmos CMA-ES e DE        
def compare_algorithms(results_df_CMA, results_df_DE, is_normal_CMA, is_normal_DE):
    """
    Compara os resultados entre os algoritmos CMA-ES e DE para cada cenário.
    """
    
    print("\n=== Comparação entre Algoritmos ===")
    
    scenarios = sorted(set(results_df_CMA['Scenario'].unique()).intersection(results_df_DE['Scenario'].unique()))
    
    for scenario in scenarios:
        data_CMA = results_df_CMA[results_df_CMA['Scenario'] == scenario]['Best_reward']
        data_DE = results_df_DE[results_df_DE['Scenario'] == scenario]['Best_reward']
        
        print(f"\nCenário: {scenario}")
        
        if is_normal_CMA.get(scenario, False) and is_normal_DE.get(scenario, False):
            # Teste t para amostras independentes (variâncias não assumidas iguais)
            stat, p_value = stats.ttest_ind(data_CMA, data_DE, equal_var=False)
            print("→ Teste t (independente)")
            print(f"Estatística t: {stat:.4f} | p-value: {p_value}")
        else:
            # Teste de Mann-Whitney U (não-paramétrico)
            stat, p_value = stats.mannwhitneyu(data_CMA, data_DE, alternative='two-sided')
            print("→ Teste de Mann-Whitney U")
            print(f"Estatística U: {stat:.4f} | p-value: {p_value}")
        
        conclusao = "Diferença significativa" if p_value < 0.05 else "Não há diferença significativa"
        print(f"Conclusão: {conclusao}")
        
# Funções para gerar gráficos comparativos entre os algoritmos CMA-ES e DE
def plots_algoritms(results_df_CMA, results_df_DE):
    """
    Gera gráficos comparativos entre os algoritmos CMA-ES e DE com melhor visualização.
    """
    
    results_df_CMA = results_df_CMA.copy()
    results_df_DE = results_df_DE.copy()
    
    results_df_CMA['Algoritmo'] = 'CMA-ES'
    results_df_DE['Algoritmo'] = 'DE'
    
    # Junta os dois dataframes
    combined_df = pd.concat([results_df_CMA, results_df_DE], ignore_index=True)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    ax = sns.boxplot(
        x='Scenario', 
        y='Best_reward', 
        hue='Algoritmo', 
        data=combined_df, 
        palette={'CMA-ES': 'royalblue', 'DE': 'darkorange'}
    )
    
    plt.title('Comparação de Algoritmos - CMA-ES vs DE', fontsize=16)
    plt.xlabel('Cenário', fontsize=14)
    plt.ylabel('Melhor reward', fontsize=14)
    plt.legend(title='Algoritmo')
    plt.tight_layout()
    plt.savefig('pComparison_algorithmsDE_CMA-ES.png')
    plt.close()
    
def plot_line_comparison(base_folder="results_seed_"):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    ALGORITHMS = ["CMA-ES", "DE"]
    SCENARIOS = ["DownStepper-v0", "ObstacleTraverser-v0"]
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14, 6), sharey=True)

    if len(SCENARIOS) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, SCENARIOS):
        legend_patches = []

        for algorithm, color in zip(ALGORITHMS, ['blue', 'orange']):
            generation_rewards = {gen: [] for gen in range(100)}
            for seed in [51, 52, 53, 54, 55]:
                folder = os.path.join(f"{base_folder}{seed}", f"NeuralController_{scenario}_{algorithm}")
                for gen in range(100):
                    gen_path = os.path.join(folder, f"generation_{gen}.csv")
                    if os.path.exists(gen_path):
                        try:
                            df = pd.read_csv(gen_path, comment="#")
                            best_reward = max(df['Reward'])
                            generation_rewards[gen].append(best_reward)
                        except Exception as e:
                            print(f"Erro ao ler {gen_path}: {e}")

            means = []
            stds = []
            maxs = []
            gens = sorted(generation_rewards.keys())
            for gen in gens:
                rewards = generation_rewards[gen]
                if rewards:
                    means.append(np.mean(rewards))
                    stds.append(np.std(rewards))
                    maxs.append(np.max(rewards))  # Máximo entre seeds
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    maxs.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)
            maxs = np.array(maxs)

            # Linha da média
            ax.plot(gens, means, label=f"{algorithm} - Média", color=color)
            ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.3)

            # Linha do máximo
            ax.plot(gens, maxs, linestyle='--', color=color, alpha=0.7, label=f"{algorithm} - Máximo")

            legend_patches.append(Patch(facecolor=color, alpha=0.3, label=f"{algorithm} ± DP"))

        ax.set_title(f'Cenário: {scenario}')
        ax.set_xlabel('Geração')
        ax.set_ylabel('Melhor reward')
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + legend_patches, labels + [p.get_label() for p in legend_patches], title="Algoritmo")

    fig.suptitle('Comparação de Algoritmos - CMA-ES vs DE', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('pcomparacao_linhas_algoritmos.png')
    plt.close()

# Função principal que executa toda a análise
def main():
    print("Iniciando análise estatística dos resultados...")
    
    ALGO = ["CMA-ES", "DE"]
    
    # Análise para cada algoritmo
    for algorithm in ALGO:
        print(f"\nAnalisando resultados para o algoritmo: {algorithm}")
        
        # Carregar resultados
        results_df = load_all_results("results_seed_", algorithm)
        
        if results_df.empty:
            print("Não foram encontrados dados para análise.")
            continue
        
        print(f"Dados carregados: {len(results_df)} entradas")
        
        # Análise descritiva
        desc_stats = descriptive_analysis(results_df)
        print(desc_stats)
        
        # Verificar normalidade
        is_normal = check_normality(results_df)
        
        # Comparar cenários
        compare_scenarios(results_df, is_normal)
        
        # Comparar seeds dentro de cada cenário
        compare_seeds(results_df, is_normal, algorithm)
        
        # Analisar progressão
        analyze_progression("results_seed_", algorithm)
        
        print(f"Análise completa para o algoritmo: {algorithm}")
        
    # Comparação entre algoritmos
    print("\n=== Comparação entre Algoritmos ===")
        
    results_df_CMA = load_all_results("results_seed_", "CMA-ES")
    results_df_DE = load_all_results("results_seed_", "DE")
    
    is_normal_CMA = check_normality(results_df_CMA)
    is_normal_DE = check_normality(results_df_DE)
        
    compare_algorithms(results_df_CMA, results_df_DE, is_normal_CMA, is_normal_DE)
    plots_algoritms(results_df_CMA, results_df_DE)
    plot_line_comparison()
    
       

if __name__ == "__main__":
    main()
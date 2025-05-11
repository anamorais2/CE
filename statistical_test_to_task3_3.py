
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Função para carregar os resultados de todas as execuções
def load_all_results(base_folder):
    """
    Carrega os resultados de todas as execuções para ambos os cenários.
    Para cada geração, coleta o melhor fitness.
    Ao final, imprime a média e o desvio padrão dos melhores fitnesss por geração para cada cenário.
    """
    all_results = []
    best_fitnesss_by_scenario = {
        'CaveCrawler-v0': [],
        'GapJumper-v0': []
    }

    # Para cada seed
    for seed in [42,43,44,45,46]:
        seed_folder = f"{base_folder}{seed}"

        # Para cada cenário
        for scenario in ['CaveCrawler-v0', 'GapJumper-v0']:
            scenario_folder = os.path.join(seed_folder, f"GA_CMAES_{scenario}")

            if not os.path.exists(scenario_folder):
                print(f"Pasta não encontrada: {scenario_folder}")
                continue

            gen_files = glob.glob(os.path.join(scenario_folder, "generation_*.csv"))
            if not gen_files:
                print(f"Nenhum arquivo de geração encontrado em: {scenario_folder}")
                continue

            best_fitnesss_by_gen = []

            for gen_file in gen_files:
                try:
                    gen_data = pd.read_csv(gen_file, comment="#")
                    best_in_gen = max(gen_data['Fitness'])
                    best_fitnesss_by_gen.append(best_in_gen)
                except Exception as e:
                    print(f"Erro ao carregar {gen_file}: {e}")

            if best_fitnesss_by_gen:
                best_fitness_overall = max(best_fitnesss_by_gen)
                all_results.append({
                    'Seed': seed,
                    'Scenario': scenario,
                    'Best_fitness': best_fitness_overall
                })
                best_fitnesss_by_scenario[scenario].extend(best_fitnesss_by_gen)

    # Print das estatísticas por cenário
    for scenario in ['CaveCrawler-v0', 'GapJumper-v0']:
        fitnesss = best_fitnesss_by_scenario[scenario]
        if fitnesss:
            avg = np.mean(fitnesss)
            std = np.std(fitnesss)
            print(f"GA_CMAES - {scenario}")
            print(f"→ Média dos melhores fitnesss de cada geração: {avg:.4f}")
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
    stats_by_scenario = results_df.groupby('Scenario')['Best_fitness'].describe()
    print(stats_by_scenario)
    
    # Cria boxplots
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Scenario', y='Best_fitness', data=results_df)
    plt.title('Distribuição de fitness por Cenário')
    plt.tight_layout()
    plt.savefig('boxplot_fitness_by_scenario.png')
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
        data = results_df[results_df['Scenario'] == scenario]['Best_fitness']
        stat, p_value = stats.shapiro(data)
        
        is_normal[scenario] = p_value > 0.05
        
        print(f"Cenário: {scenario}")
        print(f"Estatística do teste: {stat:.4f}")
        print(f"Valor-p: {p_value}")
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
    data1 = results_df[results_df['Scenario'] == scenarios[0]]['Best_fitness']
    data2 = results_df[results_df['Scenario'] == scenarios[1]]['Best_fitness']
    
    # Verifica se ambos são normais
    if is_normal.get(scenarios[0], False) and is_normal.get(scenarios[1], False):
        # Teste t para amostras independentes
        print("\n=== Teste t para Amostras Independentes ===")
        stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Usando Welch's t-test
        
        print(f"Comparando {scenarios[0]} vs {scenarios[1]}")
        print(f"Estatística t: {stat:.4f}")
        print(f"Valor-p: {p_value}")
        print(f"Conclusão: {'Diferença significativa' if p_value < 0.05 else 'Não há diferença significativa'}")
    else:
        # Teste de Mann-Whitney U (não paramétrico)
        print("\n=== Teste de Mann-Whitney U ===")
        stat, p_value = stats.mannwhitneyu(data1, data2)
        
        print(f"Comparando {scenarios[0]} vs {scenarios[1]}")
        print(f"Estatística U: {stat:.4f}")
        print(f"Valor-p: {p_value}")
        print(f"Conclusão: {'Diferença significativa' if p_value < 0.05 else 'Não há diferença significativa'}")

# Função para comparar as recompensas entre seeds
def compare_seeds(results_df, is_normal):
    """
    Compara as recompensas entre seeds para cada cenário usando ANOVA ou Kruskal-Wallis.
    """
    for scenario in results_df['Scenario'].unique():
        print(f"\n--- Comparando seeds para o cenário: {scenario} ---")

        seed_groups = []
        seed_labels = []

        for seed in results_df['Seed'].unique():
            scenario_folder = os.path.join(f"results_seed_{seed}", f"GA_CMAES_{scenario}")
            gen_path = os.path.join(scenario_folder, "generation_50.csv")

            if os.path.exists(gen_path):
                try:
                    fitnesss = extract_fitnesss(gen_path, is_normal)
                    # Flatten the list of fitness arrays
                    flat_fitnesss = np.concatenate(fitnesss)
                    seed_groups.append(flat_fitnesss)
                    seed_labels.append(f"Seed {seed}")
                except Exception as e:
                    print(f"Erro ao processar seed {seed} para cenário {scenario}: {e}")
            else:
                print(f"Arquivo não encontrado: {gen_path}")

        # Verificação de número mínimo de grupos
        if len(seed_groups) < 2:
            print("Não há grupos suficientes para comparação (mínimo 2). Pulando este cenário.")
            continue

        # Verifica se os dados são normais
        if is_normal.get(scenario, False):
            print("Usando ANOVA")
            f_stat, p_value = stats.f_oneway(*seed_groups)
            print(f"Estatística F: {f_stat:.4f} | Valor-p: {p_value}")
        else:
            print("Usando Kruskal-Wallis")
            h_stat, p_value = stats.kruskal(*seed_groups)
            print(f"Estatística H: {h_stat:.4f} | Valor-p: {p_value}")

        print(f"Conclusão: {'Diferença significativa entre seeds' if p_value < 0.05 else 'Não há diferença significativa entre seeds'}")

# Função para extrair as recompensas de todas as gerações
def extract_fitnesss(file_path, is_normal):
    """
    Extrai os valores de recompensa de todas as gerações do arquivo CSV.
    """
    df = pd.read_csv(file_path, comment="#")
    all_fitnesss = []

    for index, row in df.iterrows():
        fitness_val = row['Fitness']

        try:
            if isinstance(fitness_val, str):
                #~ Se for uma string, tenta converter para um array
                cleaned = fitness_val.strip('[]').replace('\n', '').strip()
                if cleaned:
                    fitness_array = np.fromstring(cleaned, sep=',')
                else:
                    fitness_array = np.array([])
            elif isinstance(fitness_val, (int, float)):
                fitness_array = np.array([fitness_val])
            else:
                raise TypeError(f"Tipo inesperado em Fitness: {type(fitness_val)}")

            all_fitnesss.append(fitness_array)

        except Exception as e:
            print(f"[Linha {index}] Erro ao processar Fitness '{fitness_val}': {e}")
            continue

    return all_fitnesss

# Função para análise da progressão ao longo das gerações
def analyze_progression(base_folder):
    """
    Analisa a progressão do fitness ao longo das gerações.
    """
    print("\n=== Análise da Progressão do fitness ===")
    
    # Para cada cenário
    for scenario in ['CaveCrawler-v0', 'GapJumper-v0']:
        plt.figure(figsize=(12, 8))
        
        # Para cada seed
        for seed in [42, 43, 44, 45, 46]:
            progression_data = []
            seed_folder = f"{base_folder}{seed}" 
            scenario_folder = os.path.join(seed_folder, f"GA_CMAES_{scenario}") 
            
            if not os.path.exists(scenario_folder):
                continue
                
            gen_files = glob.glob(os.path.join(scenario_folder, "generation_*.csv")) 
            
            for gen_file in sorted(gen_files, key=lambda x: int(x.split("_")[-1].split(".")[0])):
                gen_num = int(gen_file.split("_")[-1].split(".")[0])
                
                try:
                    gen_data = pd.read_csv(gen_file, comment="#")
                    # Pega a melhor Fitness da geração
                    best_fitness = max(gen_data['Fitness'])
                    progression_data.append((gen_num, best_fitness))
                except Exception as e:
                    print(f"Erro ao carregar {gen_file}: {e}")
            
            if progression_data:
                gens, fitnesss = zip(*progression_data)
                plt.plot(gens, fitnesss, label=f"Seed {seed}")
        
        plt.title(f'Progressão do fitness ao Longo das Gerações - {scenario}')
        plt.xlabel('Geração')
        plt.ylabel('Melhor fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'progression_{scenario.replace("-", "_")}_GA_CMAES.png')
        plt.close()

# Função para gerar gráfico comparativo entre cenários
def plot_line_comparison(base_folder="results_seed_"):
    """
    Gera um gráfico comparativo da evolução das recompensas nos dois cenários.
    """
    SCENARIOS = ["CaveCrawler-v0", "GapJumper-v0"]
    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 8))

    colors = ['blue', 'green']
    
    for scenario, color in zip(SCENARIOS, colors):
        generation_fitnesss = {gen: [] for gen in range(100)}
        for seed in [42, 43, 44, 45, 46]:
            folder = os.path.join(f"{base_folder}{seed}", f"GA_CMAES_{scenario}")
            for gen in range(100):
                gen_path = os.path.join(folder, f"generation_{gen}.csv")
                if os.path.exists(gen_path):
                    try:
                        df = pd.read_csv(gen_path, comment="#")
                        best_fitness = max(df['Fitness'])
                        generation_fitnesss[gen].append(best_fitness)
                    except Exception as e:
                        print(f"Erro ao ler {gen_path}: {e}")

        means = []
        stds = []
        maxs = []
        gens = sorted(generation_fitnesss.keys())
        for gen in gens:
            fitnesss = generation_fitnesss[gen]
            if fitnesss:
                means.append(np.mean(fitnesss))
                stds.append(np.std(fitnesss))
                maxs.append(np.max(fitnesss))  # Máximo entre seeds
            else:
                means.append(np.nan)
                stds.append(np.nan)
                maxs.append(np.nan)

        means = np.array(means)
        stds = np.array(stds)
        maxs = np.array(maxs)

        # Linha da média
        plt.plot(gens, means, label=f"{scenario} - Média", color=color)
        plt.fill_between(gens, means - stds, means + stds, color=color, alpha=0.3, label=f"{scenario} ± DP")

        # Linha do máximo
        plt.plot(gens, maxs, linestyle='--', color=color, alpha=0.7, label=f"{scenario} - Máximo")

    plt.title('Comparação entre Cenários - GA_CMAES', fontsize=16)
    plt.xlabel('Geração', fontsize=14)
    plt.ylabel('Melhor fitness', fontsize=14)
    plt.legend(title="Cenário")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparacao_cenarios_GA_CMAES.png')
    plt.close()

# Função principal que executa toda a análise
def main():
    print("Iniciando análise estatística dos resultados...")
    
    results_df = load_all_results("results_seed_")
    
    if results_df.empty:
        print("Não foram encontrados dados para análise.")
        return
    
    print(f"Dados carregados: {len(results_df)} entradas")
    
    # Análise descritiva
    desc_stats = descriptive_analysis(results_df)
    print(desc_stats)
    
    # Verificar normalidade
    is_normal = check_normality(results_df)
    
    # Comparar cenários
    compare_scenarios(results_df, is_normal)
    
    # Comparar seeds dentro de cada cenário
    compare_seeds(results_df, is_normal)
    
    # Analisar progressão
    analyze_progression("results_seed_")
    
    # Gerar gráfico comparativo
    plot_line_comparison()
    
    print("Análise completa para o algoritmo GA_CMAES.")

if __name__ == "__main__":
    main()
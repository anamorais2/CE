# ----------------------- IMPLEMENTAR TESTES ESTATISTICOS -----------------------
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import os
import re
import glob
import test_diversity_3_1 as td

# ------------------------ CARREGAR OS DADOS ------------------------
# DOIS ALGORITMOS: GA ROBUSTO E GA SIMPLE

# 1. Load the summary data
def load_summary_data(file_path):
   
    try:
        df = pd.read_excel(file_path, engine='openpyxl')  
    except Exception as e:
        print(f"Erro ao carregar o arquivo Excel: {e}")
        return None

    if 'Reward' not in df.columns:
        df['Reward'] = np.nan  
    

    df['Reward'] = df['Reward'].fillna(np.nan)
    

    required_columns = ['Best Fitness', 'Reward', 'Mutation Rate', 'Crossover Rate', 'Execution Time (s)']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan  
    
    # Converter colunas específicas para float, substituindo vírgulas por pontos e removendo espaços
    for col in required_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    

    return df

def load_data(base_dir):
  
    results = {}
    
    seed_pattern = re.compile(r'results_seed_(\d+)')
    
    seed_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and seed_pattern.match(d)]
    
    for seed_dir in seed_dirs:
        seed_num = seed_pattern.match(seed_dir).group(1)
        seed_path = os.path.join(base_dir, seed_dir)
        
        results[seed_num] = {}
        
        controller_dirs = [d for d in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, d))]
        
        for controller_dir in controller_dirs:
            alg_type = "Simple_GA" if "Simple" in controller_dir else "GA"
            
            if "GA_CMAES" in controller_dir:
                continue
            
            parts = controller_dir.split('_')
            
           
            controller_type = parts[0] 
            
        
            if "Simple" in controller_dir:
                
                scenario = '_'.join([p for p in parts[2:-1] if p != "Simple"])
            else:
             
                scenario = '_'.join(parts[2:])
            
            # Estrutura: results[seed][controller_type][scenario][alg_type]
            if controller_type not in results[seed_num]:
                results[seed_num][controller_type] = {}
            
            if scenario not in results[seed_num][controller_type]:
                results[seed_num][controller_type][scenario] = {}
            
       
            controller_path = os.path.join(seed_path, controller_dir)
            generation_files = sorted(glob.glob(os.path.join(controller_path, "generation_*.csv")))
            
           
            
            if not generation_files:
                print(f"Nenhum arquivo de geração encontrado para {controller_dir} na seed {seed_num}")
                continue
            
            # Carregar metadados do primeiro arquivo de geração
            metadata = {}
            first_file = generation_files[0]
            with open(first_file, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        break
           
                    line = line.strip('# \n')
                    if ',' in line:
                        key, value = line.split(',', 1)
                        metadata[key] = value
            
            all_data = []
            
            for gen_file in generation_files:
       
                gen_match = re.search(r'[Gg]eneration_(\d+)', gen_file)
                if gen_match:
                    gen_num = int(gen_match.group(1))
                else:
    
                    continue
                
                try:
               
                    if gen_num == 0:
                        # Para a geração 0, pular todas as linhas que começam com #
                        df = pd.read_csv(gen_file, comment='#')
                    else:
                        # Para outras gerações, pular apenas a primeira linha (cabeçalho)
                        df = pd.read_csv(gen_file)
           
                    df['Generation'] = gen_num
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"Erro ao processar arquivo {gen_file}: {e}")
            
            if all_data:
     
                combined_df = pd.concat(all_data, ignore_index=True)
                
                for key, value in metadata.items():
                    combined_df[key] = value
                
              
                combined_df['Controller'] = controller_type
                combined_df['Scenario'] = scenario
                combined_df['SEED'] = seed_num
                combined_df['AlgType'] = alg_type
                
            
                results[seed_num][controller_type][scenario][alg_type] = combined_df
    
    return results


# Função auxiliar para verificar a normalidade dos dados
def check_normality(data, column='Best Fitness', alpha=0.05):
    
  
    values = data[column].dropna()
    

    if len(values) < 3:
        print("Dados insuficientes para teste de normalidade")
        return False, None
    
    # Teste de Shapiro-Wilk
    stat, p_value = stats.shapiro(values)
    is_normal = p_value > alpha
    
    # Exibir resultados
    result = "normalmente distribuídos" if is_normal else "não normalmente distribuídos"
    print(f"Teste de Shapiro-Wilk para {column}: p-value = {p_value:.4f} ({result})")
    
    return is_normal, p_value


# Função para extrair o último valor de fitness/reward para cada seed
def extract_final_performance(results, controller, scenario, alg_type, metric='Best Fitness'):

    final_values = []
    
    for seed in results.keys():
        try:
            data = results[seed][controller][scenario][alg_type]
            last_gen = data['Generation'].max()
            final_data = data[data['Generation'] == last_gen]
            
            # Usar a média dos valores da última geração para cada seed
            if alg_type == 'Simple_GA' and metric == 'Reward':
                # Para o SimpleGA, usar Fitness quando se pede Reward
                final_values.append(final_data['Fitness'].mean())
            else:
                final_values.append(final_data[metric].mean())
                
        except KeyError:
            pass
    
    return final_values


# 1.1 Comparação entre Cenários (mesmo controlador e algoritmo)
def compare_scenarios_paired(results, controllers, scenarios, alg_types, metric='Reward'):
   
    # Compara dois cenários diferentes para o mesmo controlador e algoritmo usando teste dependente.
  
    
    if len(scenarios) != 2:
        print("Erro: Esta função requer exatamente 2 cenários para comparação")
        return
    
    scenario1, scenario2 = scenarios
    
    print(f"\n{'='*80}")
    print(f"1.1 COMPARAÇÃO ENTRE CENÁRIOS: {scenario1} vs {scenario2}")
    print(f"{'='*80}")
    
    for controller in controllers:
        print(f"\nControlador: {controller}")
        print(f"{'-'*50}")
        
        for alg_type in alg_types:
            print(f"\nAlgoritmo: {alg_type}")
            
            # Obter valores finais para cada cenário
            values_scenario1 = extract_final_performance(results, controller, scenario1, alg_type, metric)
            values_scenario2 = extract_final_performance(results, controller, scenario2, alg_type, metric)
            
            # Verificar se há dados suficientes
            if len(values_scenario1) < 2 or len(values_scenario2) < 2:
                print(f"  Dados insuficientes para comparação ({len(values_scenario1)} vs {len(values_scenario2)} amostras)")
                continue
            
            # Verificar se temos o mesmo número de seeds para comparação dependente
            if len(values_scenario1) != len(values_scenario2):
                # Se não for mesmo número, fazer teste não dependente
                print(f"  Aviso: Número diferente de seeds ({len(values_scenario1)} vs {len(values_scenario2)}). Usando teste não pareado.")
                
                # Estatísticas descritivas
                mean1, std1 = np.mean(values_scenario1), np.std(values_scenario1)
                mean2, std2 = np.mean(values_scenario2), np.std(values_scenario2)
                
                print(f"  {scenario1}: {mean1:.4f} ± {std1:.4f} ({len(values_scenario1)} seeds)")
                print(f"  {scenario2}: {mean2:.4f} ± {std2:.4f} ({len(values_scenario2)} seeds)")
                
                # Verificar normalidade
                is_normal1 = stats.shapiro(values_scenario1)[1] > 0.05 if len(values_scenario1) >= 3 else False
                is_normal2 = stats.shapiro(values_scenario2)[1] > 0.05 if len(values_scenario2) >= 3 else False
                
                if is_normal1 and is_normal2:
                    # Teste t independente
                    t_stat, p_value = stats.ttest_ind(values_scenario1, values_scenario2, equal_var=False)
                    test_name = "Ind T - Teste"
                else:
                    # Mann-Whitney
                    u_stat, p_value = stats.mannwhitneyu(values_scenario1, values_scenario2, alternative='two-sided')
                    test_name = "Mann-Whitney U"
            else:
                # Temos o mesmo número de seeds: fazer teste dependente
                mean1, std1 = np.mean(values_scenario1), np.std(values_scenario1)
                mean2, std2 = np.mean(values_scenario2), np.std(values_scenario2)
                
                print(f"  {scenario1}: {mean1:.4f} ± {std1:.4f} ({len(values_scenario1)} seeds)")
                print(f"  {scenario2}: {mean2:.4f} ± {std2:.4f} ({len(values_scenario2)} seeds)")
                
                # Verificar normalidade das diferenças para teste dependente
                differences = np.array(values_scenario1) - np.array(values_scenario2)
                is_normal = stats.shapiro(differences)[1] > 0.05 if len(differences) >= 3 else False
                
                if is_normal:
                    # Teste t dependente
                    t_stat, p_value = stats.ttest_rel(values_scenario1, values_scenario2)
                    test_name = "Dep T - Test "
                else:
                    # Wilcoxon signed-rank
                    w_stat, p_value = stats.wilcoxon(values_scenario1, values_scenario2)
                    test_name = "Wilcoxon"
            
  
            print(f"  {test_name}: p-value = {p_value:.4f}")
            
            alpha = 0.05
            if p_value < alpha:
                if mean1 > mean2:
                    print(f"  Conclusão: {scenario1} apresenta desempenho significativamente superior (p < {alpha})")
                else:
                    print(f"  Conclusão: {scenario2} apresenta desempenho significativamente superior (p < {alpha})")
            else:
                print(f"  Conclusão: Não há diferença estatisticamente significativa entre os cenários (p > {alpha})")
            
          
            plt.figure(figsize=(10, 6))
            
         
            data = [values_scenario1, values_scenario2]
            labels = [scenario1, scenario2]
            
            bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)            
        
            for i, d in enumerate(data):
                x = np.random.normal(i+1, 0.1, size=len(d))
                plt.scatter(x, d, alpha=0.6)
            
            plt.title(f'Comparação entre Cenários: {controller}, {alg_type}')
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()


# 1.2 Comparação entre Controladores (mesmo cenário e algoritmo)
def compare_controllers(results, controllers, scenarios, alg_types, metric='Reward'):
    
    # Compara diferentes controladores no mesmo cenário e algoritmo.
    
  
    print(f"\n{'='*80}")
    print(f"1.2 COMPARAÇÃO ENTRE CONTROLADORES")
    print(f"{'='*80}")
    
    for scenario in scenarios:
        print(f"\nCenário: {scenario}")
        print(f"{'-'*50}")
        
        for alg_type in alg_types:
            print(f"\nAlgoritmo: {alg_type}")
            
    
            controllers_data = {}
            for controller in controllers:
                values = extract_final_performance(results, controller, scenario, alg_type, metric)
                if len(values) > 0:
                    controllers_data[controller] = values
            
    
            if len(controllers_data) < 2:
                print(f"  Dados insuficientes para comparação (apenas {len(controllers_data)} controladores disponíveis)")
                continue
            
            for controller, values in controllers_data.items():
                mean, std = np.mean(values), np.std(values)
                print(f"  {controller}: {mean:.4f} ± {std:.4f} ({len(values)} seeds)")
            
    
            all_values = []
            labels = []
            
            for controller, values in controllers_data.items():
                all_values.extend(values)
                labels.extend([controller] * len(values))
            
            df = pd.DataFrame({
                'Value': all_values,
                'Controller': labels
            })
            
   
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Controller', y='Value', data=df)
            
    
            sns.stripplot(x='Controller', y='Value', data=df, 
                         color='black', alpha=0.5, jitter=True)
            
            plt.title(f'Comparação entre Controladores: {scenario}, {alg_type}')
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
         
            if len(controllers_data) == 2:
                # Para 2 controladores: teste t ou Mann-Whitney
                controller1, controller2 = list(controllers_data.keys())
                values1, values2 = controllers_data[controller1], controllers_data[controller2]
                
                # Verificar normalidade
                is_normal1 = stats.shapiro(values1)[1] > 0.05 if len(values1) >= 3 else False
                is_normal2 = stats.shapiro(values2)[1] > 0.05 if len(values2) >= 3 else False
                
                if is_normal1 and is_normal2:
                    # Teste t
                    t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                    test_name = "Teste t de Welch"
                else:
                    # Mann-Whitney
                    u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                
                print(f"\n  {test_name}: p-value = {p_value:.4f}")
                
                alpha = 0.05
                if p_value < alpha:
                    mean1, mean2 = np.mean(values1), np.mean(values2)
                    if mean1 > mean2:
                        print(f"  Conclusão: {controller1} apresenta desempenho significativamente superior (p < {alpha})")
                    else:
                        print(f"  Conclusão: {controller2} apresenta desempenho significativamente superior (p < {alpha})")
                else:
                    print(f"  Conclusão: Não há diferença estatisticamente significativa entre os controladores (p > {alpha})")
                
            else:
                # Para 3+ controladores: ANOVA ou Kruskal-Wallis
                # Verificar normalidade e homogeneidade de variâncias
                use_parametric = True
                for values in controllers_data.values():
                    if len(values) >= 3:
                        if stats.shapiro(values)[1] <= 0.05:
                            use_parametric = False
                            break
                
                if use_parametric:

                    # ANOVA
                    f_stat, p_value = stats.f_oneway(*controllers_data.values())
                    test_name = "ANOVA"
                    
                else:
                    # Kruskal-Wallis
                    h_stat, p_value = stats.kruskal(*controllers_data.values())
                    test_name = "Kruskal-Wallis"
                
                print(f"\n  {test_name}: p-value = {p_value:.4f}")
                
                alpha = 0.05
                if p_value < alpha:
                    print(f"  Conclusão: Há diferença estatisticamente significativa entre pelo menos dois controladores (p < {alpha})")
                    
                else:
                    print(f"  Conclusão: Não há diferença estatisticamente significativa entre os controladores (p > {alpha})")


# 1.3 Comparação entre Algoritmos (mesmo controlador e cenário)
def compare_algorithms_paired(results, controllers, scenarios, metric='Reward'):

    # Compara dois algoritmos (GA vs Simple_GA) para o mesmo controlador e cenário.
    
  
    print(f"\n{'='*80}")
    print(f"1.3 COMPARAÇÃO ENTRE ALGORITMOS: GA vs Simple_GA")
    print(f"{'='*80}")
    
    for controller in controllers:
        for scenario in scenarios:
            print(f"\nControlador: {controller}, Cenário: {scenario}")
            print(f"{'-'*50}")
            
            # Extrair dados para GA
            ga_values = extract_final_performance(results, controller, scenario, 'GA', metric)
            
            # Para Simple_GA, adaptar métrica (Simple_GA usa 'Fitness' em vez de 'Reward')
            simple_ga_metric = 'Fitness' if metric == 'Reward' else metric
            simple_ga_values = extract_final_performance(results, controller, scenario, 'Simple_GA', simple_ga_metric)
            
            # Verificar se há dados suficientes
            if len(ga_values) < 2 or len(simple_ga_values) < 2:
                print(f"  Dados insuficientes para comparação ({len(ga_values)} vs {len(simple_ga_values)} amostras)")
                continue
            

            ga_mean, ga_std = np.mean(ga_values), np.std(ga_values)
            simple_mean, simple_std = np.mean(simple_ga_values), np.std(simple_ga_values)
            
            print(f"  GA: {ga_mean:.4f} ± {ga_std:.4f} ({len(ga_values)} seeds)")
            print(f"  Simple_GA: {simple_mean:.4f} ± {simple_std:.4f} ({len(simple_ga_values)} seeds)")
            
            # Verificar se temos o mesmo número de seeds para comparação pareada
            if len(ga_values) != len(simple_ga_values):
                # Se não for mesmo número, fazer teste independente
                print(f"  Aviso: Número diferente de seeds ({len(ga_values)} vs {len(simple_ga_values)}). Usando teste não pareado.")
                
                # Verificar normalidade
                ga_normal = stats.shapiro(ga_values)[1] > 0.05 if len(ga_values) >= 3 else False
                simple_normal = stats.shapiro(simple_ga_values)[1] > 0.05 if len(simple_ga_values) >= 3 else False
                
                if ga_normal and simple_normal:
                    # Teste t independente
                    t_stat, p_value = stats.ttest_ind(ga_values, simple_ga_values, equal_var=False)
                    test_name = "Teste t de Welch (não pareado)"
                else:
                    # Mann-Whitney
                    u_stat, p_value = stats.mannwhitneyu(ga_values, simple_ga_values, alternative='two-sided')
                    test_name = "Mann-Whitney U (não pareado)"
            else:
                # Temos o mesmo número de seeds: fazer teste dependente
                # Verificar normalidade das diferenças para teste dependente
                differences = np.array(ga_values) - np.array(simple_ga_values)
                is_normal = stats.shapiro(differences)[1] > 0.05 if len(differences) >= 3 else False
                
                if is_normal:
                    # Teste t dependente
                    t_stat, p_value = stats.ttest_rel(ga_values, simple_ga_values)
                    test_name = "Teste t pareado"
                else:
                    # Wilcoxon signed-rank
                    w_stat, p_value = stats.wilcoxon(ga_values, simple_ga_values)
                    test_name = "Wilcoxon signed-rank (pareado)"
            
            print(f"  {test_name}: p-value = {p_value:.4f}")
            
            alpha = 0.05
            if p_value < alpha:
                if ga_mean > simple_mean:
                    print(f"  Conclusão: GA apresenta desempenho significativamente superior (p < {alpha})")
                else:
                    print(f"  Conclusão: Simple_GA apresenta desempenho significativamente superior (p < {alpha})")
            else:
                print(f"  Conclusão: Não há diferença estatisticamente significativa entre os algoritmos (p > {alpha})")
            
        
            plt.figure(figsize=(10, 6))
            
           
            data = [ga_values, simple_ga_values]
            labels = ['GA', 'Simple_GA']
            
            bp = plt.boxplot(data, tick_labels=labels, patch_artist=True)
         
            for i, d in enumerate(data):
                x = np.random.normal(i+1, 0.1, size=len(d))
                plt.scatter(x, d, alpha=0.6)
            
            plt.title(f'Comparação entre Algoritmos: {controller}, {scenario}')
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()


def analyze_convergence_dual(results, controller, scenario):
    
    # Analisa a convergência dos algoritmos comparando média geral vs média dos máximos.
    
   
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE CONVERGÊNCIA DUAL - {controller}, {scenario}")
    print(f"{'='*80}")
    
    ga_avg_by_gen = {}      # geração -> lista de médias (uma por seed)
    ga_max_by_gen = {}      # geração -> lista de máximos (um por seed)
    simple_ga_avg_by_gen = {}
    simple_ga_max_by_gen = {}
    
    for seed in results.keys():
        # GA dados
        try:
            ga_data = results[seed][controller][scenario]['GA']
            
            # Agrupar por geração
            for gen in sorted(ga_data['Generation'].unique()):
                gen_data = ga_data[ga_data['Generation'] == gen]
                
                # Média da população para esta geração e seed
                if gen not in ga_avg_by_gen:
                    ga_avg_by_gen[gen] = []
                ga_avg_by_gen[gen].append(gen_data['Reward'].mean())
                
                # Máximo da população para esta geração e seed
                if gen not in ga_max_by_gen:
                    ga_max_by_gen[gen] = []
                ga_max_by_gen[gen].append(gen_data['Reward'].max())
                
        except KeyError:
            pass
        
        # Simple GA dados
        try:
            simple_ga_data = results[seed][controller][scenario]['Simple_GA']
            
            # Agrupar por geração
            for gen in sorted(simple_ga_data['Generation'].unique()):
                gen_data = simple_ga_data[simple_ga_data['Generation'] == gen]
                
                # Média da população
                if gen not in simple_ga_avg_by_gen:
                    simple_ga_avg_by_gen[gen] = []
                simple_ga_avg_by_gen[gen].append(gen_data['Fitness'].mean())
                
                # Máximo da população
                if gen not in simple_ga_max_by_gen:
                    simple_ga_max_by_gen[gen] = []
                simple_ga_max_by_gen[gen].append(gen_data['Fitness'].max())
                
        except KeyError:
            pass
    
    # Verificar se há dados suficientes
    if not (ga_avg_by_gen or simple_ga_avg_by_gen):
        print("Dados insuficientes para análise de convergência")
        return
    
    # ----- Gráfico 1: GA - Média vs Máximo -----
    if ga_avg_by_gen:
        plt.figure(figsize=(12, 8))
        
        # Processar médias gerais
        ga_gens = sorted(ga_avg_by_gen.keys())
        ga_avg_means = [np.mean(ga_avg_by_gen[gen]) for gen in ga_gens]
        ga_avg_stds = [np.std(ga_avg_by_gen[gen]) for gen in ga_gens]
        
        # Processar médias dos máximos
        ga_max_means = [np.mean(ga_max_by_gen[gen]) for gen in ga_gens]
        ga_max_stds = [np.std(ga_max_by_gen[gen]) for gen in ga_gens]
        
        # Plotar médias
        plt.plot(ga_gens, ga_avg_means, 'b-', linewidth=2, label='GA - Média da População')
        plt.fill_between(ga_gens, 
                        [m - s for m, s in zip(ga_avg_means, ga_avg_stds)],
                        [m + s for m, s in zip(ga_avg_means, ga_avg_stds)],
                        color='blue', alpha=0.2)
        
        # Plotar máximos
        plt.plot(ga_gens, ga_max_means, 'g-', linewidth=2, label='GA - Média dos Máximos')
        plt.fill_between(ga_gens, 
                        [m - s for m, s in zip(ga_max_means, ga_max_stds)],
                        [m + s for m, s in zip(ga_max_means, ga_max_stds)],
                        color='green', alpha=0.2)
        

        plt.title(f'GA: Média da População vs Média dos Máximos - {controller}, {scenario}')
        plt.xlabel('Geração')
        plt.ylabel('Fitness/Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        

        last_gen = max(ga_gens)
        print(f"GA - Geração Final ({last_gen}):")
        print(f"  Média da População: {ga_avg_means[-1]:.4f} ± {ga_avg_stds[-1]:.4f}")
        print(f"  Média dos Máximos: {ga_max_means[-1]:.4f} ± {ga_max_stds[-1]:.4f}")
        print(f"  Diferença (Máx - Média): {ga_max_means[-1] - ga_avg_means[-1]:.4f}")

        if len(ga_gens) > 1:
            avg_improvement = (ga_avg_means[-1] - ga_avg_means[0]) / len(ga_gens)
            max_improvement = (ga_max_means[-1] - ga_max_means[0]) / len(ga_gens)
            print(f"  Taxa média de melhoria por geração (População): {avg_improvement:.4f}")
            print(f"  Taxa média de melhoria por geração (Máximos): {max_improvement:.4f}")
    
    # ----- Gráfico 2: Simple GA - Média vs Máximo -----
    if simple_ga_avg_by_gen:
        plt.figure(figsize=(12, 8))
        
        # Processar médias
        simple_ga_gens = sorted(simple_ga_avg_by_gen.keys())
        simple_ga_avg_means = [np.mean(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
        simple_ga_avg_stds = [np.std(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
        
        # Processar máximos
        simple_ga_max_means = [np.mean(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
        simple_ga_max_stds = [np.std(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
        
        # Plotar médias
        plt.plot(simple_ga_gens, simple_ga_avg_means, 'r-', linewidth=2, label='Simple GA - Média da População')
        plt.fill_between(simple_ga_gens, 
                        [m - s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                        [m + s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                        color='red', alpha=0.2)
        
        # Plotar máximos
        plt.plot(simple_ga_gens, simple_ga_max_means, 'm-', linewidth=2, label='Simple GA - Média dos Máximos')
        plt.fill_between(simple_ga_gens, 
                        [m - s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                        [m + s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                        color='magenta', alpha=0.2)
        
  
        plt.title(f'Simple GA: Média da População vs Média dos Máximos - {controller}, {scenario}')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

        last_gen = max(simple_ga_gens)
        print(f"\nSimple GA - Geração Final ({last_gen}):")
        print(f"  Média da População: {simple_ga_avg_means[-1]:.4f} ± {simple_ga_avg_stds[-1]:.4f}")
        print(f"  Média dos Máximos: {simple_ga_max_means[-1]:.4f} ± {simple_ga_max_stds[-1]:.4f}")
        print(f"  Diferença (Máx - Média): {simple_ga_max_means[-1] - simple_ga_avg_means[-1]:.4f}")
        
    
        if len(simple_ga_gens) > 1:
            avg_improvement = (simple_ga_avg_means[-1] - simple_ga_avg_means[0]) / len(simple_ga_gens)
            max_improvement = (simple_ga_max_means[-1] - simple_ga_max_means[0]) / len(simple_ga_gens)
            print(f"  Taxa média de melhoria por geração (População): {avg_improvement:.4f}")
            print(f"  Taxa média de melhoria por geração (Máximos): {max_improvement:.4f}")
    
    # ----- Gráfico 3: Comparativo dos Máximos entre algoritmos -----
    if ga_max_by_gen and simple_ga_max_by_gen:
        plt.figure(figsize=(12, 8))
        
        # Plotar GA máximos
        plt.plot(ga_gens, ga_max_means, 'g-', linewidth=2, label='GA - Média dos Máximos')
        plt.fill_between(ga_gens, 
                        [m - s for m, s in zip(ga_max_means, ga_max_stds)],
                        [m + s for m, s in zip(ga_max_means, ga_max_stds)],
                        color='green', alpha=0.2)
        
        # Plotar Simple GA máximos
        plt.plot(simple_ga_gens, simple_ga_max_means, 'm-', linewidth=2, label='Simple GA - Média dos Máximos')
        plt.fill_between(simple_ga_gens, 
                        [m - s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                        [m + s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                        color='magenta', alpha=0.2)
        
        # Configurar gráfico
        plt.title(f'Comparação da Média dos Máximos - {controller}, {scenario}')
        plt.xlabel('Geração')
        plt.ylabel('Fitness/Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
                    
# 1. Análise entre seeds para um algoritmo/controlador específico
def analyze_seeds_variation(results, controller, scenario, alg_type, metric='Best Fitness'):
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE VARIAÇÃO ENTRE SEEDS - {controller}, {scenario}, {alg_type}")
    print(f"{'='*80}")
    

    seed_data = []
    seed_groups = []
    
    for seed in results.keys():
        try:
           
            data = results[seed][controller][scenario][alg_type]
            
            # Obter a última geração (melhor fitness final)
            last_gen = data['Generation'].max()
            final_data = data[data['Generation'] == last_gen]
            
            # Adicionar ao conjunto de dados
            seed_data.extend(final_data[metric].tolist())
            seed_groups.extend([f"Seed {seed}"] * len(final_data))
            
            print(f"Seed {seed}: {len(final_data)} amostras, Média = {final_data[metric].mean():.4f}")
            
        except KeyError:
            print(f"Dados não disponíveis para Seed {seed}, {controller}, {scenario}, {alg_type}")
    

    if len(seed_data) < 2:
        print("Dados insuficientes para análise estatística")
        return None
    

    df = pd.DataFrame({
        'Value': seed_data,
        'Seed': seed_groups
    })
    
    # Visualizar a distribuição por seed
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Seed', y='Value', data=df)
    plt.title(f'Distribuição de {metric} por Seed - {controller}, {scenario}, {alg_type}')
    plt.tight_layout()
    plt.show()
    
    # Verificar normalidade agregada
    is_normal, _ = check_normality(df, column='Value')
    
    # Realizar ANOVA ou Kruskal-Wallis conforme apropriado
    unique_seeds = df['Seed'].unique()
    
    if len(unique_seeds) < 2:
        print("Apenas uma seed disponível, análise de variância impossível")
        return df
    
    if is_normal:
        # ANOVA para dados normais
        result = stats.f_oneway(*[df[df['Seed'] == seed]['Value'].values for seed in unique_seeds])
        test_name = "ANOVA"
    else:
        # Kruskal-Wallis para dados não normais
        result = stats.kruskal(*[df[df['Seed'] == seed]['Value'].values for seed in unique_seeds])
        test_name = "Kruskal-Wallis"
    
    print(f"\nResultado do teste {test_name}: estatística = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")
    alpha = 0.05
    if result.pvalue < alpha:
        print(f"Conclusão: Há diferença estatisticamente significativa entre as seeds (p < {alpha})")
        
    else:
        print(f"Conclusão: Não há diferença estatisticamente significativa entre as seeds (p > {alpha})")
    
    return df
      
                    
# Função principal para executar todas as comparações solicitadas
def run_comprehensive_analysis(results, controllers, scenarios):
 
    print("\n\n")
    print(f"{'#'*100}")
    print(f"ANÁLISE ESTATÍSTICA COMPARATIVA COMPLETA")
    print(f"{'#'*100}")

    alg_types = ['GA', 'Simple_GA']
    

    metrics = ['Reward']
    
    for metric in metrics:
        print(f"\n\n{'*'*80}")
        print(f"ANÁLISES PARA MÉTRICA: {metric}")
        print(f"{'*'*80}")
        
        # 1.1 Comparação entre Cenários
        compare_scenarios_paired(results, controllers, scenarios, alg_types, metric)
        
        # 1.2 Comparação entre Controladores
        compare_controllers(results, controllers, scenarios, alg_types, metric)
        
        # 1.3 Comparação entre Algoritmos
        compare_algorithms_paired(results, controllers, scenarios, metric)
        
    for controller in controllers:
            for scenario in scenarios:
                # Para GA
                analyze_seeds_variation(results, controller, scenario, 'GA', metric=metric)
                
                # Para Simple GA
                simple_ga_metric = 'Fitness' if metric == 'Reward' else metric
                analyze_seeds_variation(results, controller, scenario, 'Simple_GA', metric= simple_ga_metric)
                
                analyze_convergence_dual(results, controller, scenario)
        
        
def analyze_convergence_dual_combined(results, controllers, scenarios):
    """
    Analisa a convergência dos algoritmos comparando todos os controladores e cenários em um único gráfico.
    
    Args:
        results: Dicionário de resultados por seed
        controllers: Lista de nomes dos controladores a serem analisados
        scenarios: Lista de nomes dos cenários a serem analisados
    """
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE CONVERGÊNCIA COMBINADA - Todos controladores e cenários")
    print(f"{'='*80}")
    
    plt.figure(figsize=(16, 10))
    
    # Cores para cada combinação de controlador/cenário
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(controllers) * len(scenarios) * 4, 20)))
    color_idx = 0
    
    # Para cada controlador e cenário
    for controller in controllers:
        for scenario in scenarios:
            # Estruturas para armazenar dados
            ga_avg_by_gen = {}      # geração -> lista de médias (uma por seed)
            ga_max_by_gen = {}      # geração -> lista de máximos (um por seed)
            simple_ga_avg_by_gen = {}
            simple_ga_max_by_gen = {}
            
            # Coletar dados de todas as seeds
            for seed in results.keys():
                # GA dados
                try:
                    ga_data = results[seed][controller][scenario]['GA']
                    
                    # Agrupar por geração
                    for gen in sorted(ga_data['Generation'].unique()):
                        gen_data = ga_data[ga_data['Generation'] == gen]
                        
                        # Média da população para esta geração e seed
                        if gen not in ga_avg_by_gen:
                            ga_avg_by_gen[gen] = []
                        ga_avg_by_gen[gen].append(gen_data['Reward'].mean())
                        
                        # Máximo da população para esta geração e seed
                        if gen not in ga_max_by_gen:
                            ga_max_by_gen[gen] = []
                        ga_max_by_gen[gen].append(gen_data['Reward'].max())
                        
                except KeyError:
                    pass
                
                # Simple GA dados
                try:
                    simple_ga_data = results[seed][controller][scenario]['Simple_GA']
                    
                    # Agrupar por geração
                    for gen in sorted(simple_ga_data['Generation'].unique()):
                        gen_data = simple_ga_data[simple_ga_data['Generation'] == gen]
                        
                        # Média da população
                        if gen not in simple_ga_avg_by_gen:
                            simple_ga_avg_by_gen[gen] = []
                        simple_ga_avg_by_gen[gen].append(gen_data['Fitness'].mean())
                        
                        # Máximo da população
                        if gen not in simple_ga_max_by_gen:
                            simple_ga_max_by_gen[gen] = []
                        simple_ga_max_by_gen[gen].append(gen_data['Fitness'].max())
                        
                except KeyError:
                    pass
            
            # Plotar GA - Média da População
            if ga_avg_by_gen:
                ga_gens = sorted(ga_avg_by_gen.keys())
                ga_avg_means = [np.mean(ga_avg_by_gen[gen]) for gen in ga_gens]
                ga_avg_stds = [np.std(ga_avg_by_gen[gen]) for gen in ga_gens]
                
                plt.plot(ga_gens, ga_avg_means, color=colors[color_idx % len(colors)], linewidth=2, 
                         linestyle='-', label=f'{controller}, {scenario} - GA Média')
                plt.fill_between(ga_gens, 
                                [m - s for m, s in zip(ga_avg_means, ga_avg_stds)],
                                [m + s for m, s in zip(ga_avg_means, ga_avg_stds)],
                                color=colors[color_idx % len(colors)], alpha=0.1)
                color_idx += 1
                
                # Plotar GA - Média dos Máximos
                ga_max_means = [np.mean(ga_max_by_gen[gen]) for gen in ga_gens]
                ga_max_stds = [np.std(ga_max_by_gen[gen]) for gen in ga_gens]
                
                plt.plot(ga_gens, ga_max_means, color=colors[color_idx % len(colors)], linewidth=2, 
                         linestyle='--', label=f'{controller}, {scenario} - GA Máx')
                plt.fill_between(ga_gens, 
                                [m - s for m, s in zip(ga_max_means, ga_max_stds)],
                                [m + s for m, s in zip(ga_max_means, ga_max_stds)],
                                color=colors[color_idx % len(colors)], alpha=0.1)
                color_idx += 1
            
            # Plotar Simple GA - Média da População
            if simple_ga_avg_by_gen:
                simple_ga_gens = sorted(simple_ga_avg_by_gen.keys())
                simple_ga_avg_means = [np.mean(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
                simple_ga_avg_stds = [np.std(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
                
                plt.plot(simple_ga_gens, simple_ga_avg_means, color=colors[color_idx % len(colors)], linewidth=2, 
                         linestyle='-', label=f'{controller}, {scenario} - Simple GA Média')
                plt.fill_between(simple_ga_gens, 
                                [m - s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                                [m + s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                                color=colors[color_idx % len(colors)], alpha=0.1)
                color_idx += 1
                
                # Plotar Simple GA - Média dos Máximos
                simple_ga_max_means = [np.mean(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
                simple_ga_max_stds = [np.std(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
                
                plt.plot(simple_ga_gens, simple_ga_max_means, color=colors[color_idx % len(colors)], linewidth=2, 
                         linestyle='--', label=f'{controller}, {scenario} - Simple GA Máx')
                plt.fill_between(simple_ga_gens, 
                                [m - s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                                [m + s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                                color=colors[color_idx % len(colors)], alpha=0.1)
                color_idx += 1
    
    # Configurar gráfico
    plt.title('Análise de Convergência Combinada - Todos Controladores e Cenários')
    plt.xlabel('Geração')
    plt.ylabel('Fitness/Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar legenda para não ficar sobreposta
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

def plot_algorithm_specific_graphs(results, controllers, scenarios):
    """
    Gera gráficos separados por tipo de algoritmo para facilitar a visualização.
    
    Args:
        results: Dicionário de resultados por seed
        controllers: Lista de nomes dos controladores a serem analisados
        scenarios: Lista de nomes dos cenários a serem analisados
    """
    # Gráfico para GA - Apenas Médias
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers) * len(scenarios)))
    color_idx = 0
    
    for controller in controllers:
        for scenario in scenarios:
            ga_avg_by_gen = {}
            
            # Coletar dados de todas as seeds
            for seed in results.keys():
                try:
                    ga_data = results[seed][controller][scenario]['GA']
                    
                    # Agrupar por geração
                    for gen in sorted(ga_data['Generation'].unique()):
                        gen_data = ga_data[ga_data['Generation'] == gen]
                        
                        # Média da população para esta geração e seed
                        if gen not in ga_avg_by_gen:
                            ga_avg_by_gen[gen] = []
                        ga_avg_by_gen[gen].append(gen_data['Reward'].mean())
                        
                except KeyError:
                    pass
            
            # Plotar GA - Média da População
            if ga_avg_by_gen:
                ga_gens = sorted(ga_avg_by_gen.keys())
                ga_avg_means = [np.mean(ga_avg_by_gen[gen]) for gen in ga_gens]
                ga_avg_stds = [np.std(ga_avg_by_gen[gen]) for gen in ga_gens]
                
                plt.plot(ga_gens, ga_avg_means, color=colors[color_idx], linewidth=2, 
                         marker='o', markersize=4, label=f'{controller}, {scenario}')
                plt.fill_between(ga_gens, 
                                [m - s for m, s in zip(ga_avg_means, ga_avg_stds)],
                                [m + s for m, s in zip(ga_avg_means, ga_avg_stds)],
                                color=colors[color_idx], alpha=0.2)
                color_idx += 1
    
    plt.title('GA - Média da População: Comparação entre Controladores e Cenários')
    plt.xlabel('Geração')
    plt.ylabel('Fitness/Reward Médio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Gráfico para GA - Apenas Máximos
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers) * len(scenarios)))
    color_idx = 0
    
    for controller in controllers:
        for scenario in scenarios:
            ga_max_by_gen = {}
            
            # Coletar dados de todas as seeds
            for seed in results.keys():
                try:
                    ga_data = results[seed][controller][scenario]['GA']
                    
                    # Agrupar por geração
                    for gen in sorted(ga_data['Generation'].unique()):
                        gen_data = ga_data[ga_data['Generation'] == gen]
                        
                        # Máximo da população para esta geração e seed
                        if gen not in ga_max_by_gen:
                            ga_max_by_gen[gen] = []
                        ga_max_by_gen[gen].append(gen_data['Reward'].max())
                        
                except KeyError:
                    pass
            
            # Plotar GA - Média dos Máximos
            if ga_max_by_gen:
                ga_gens = sorted(ga_max_by_gen.keys())
                ga_max_means = [np.mean(ga_max_by_gen[gen]) for gen in ga_gens]
                ga_max_stds = [np.std(ga_max_by_gen[gen]) for gen in ga_gens]
                
                plt.plot(ga_gens, ga_max_means, color=colors[color_idx], linewidth=2, 
                         marker='s', markersize=4, label=f'{controller}, {scenario}')
                plt.fill_between(ga_gens, 
                                [m - s for m, s in zip(ga_max_means, ga_max_stds)],
                                [m + s for m, s in zip(ga_max_means, ga_max_stds)],
                                color=colors[color_idx], alpha=0.2)
                color_idx += 1
    
    plt.title('GA - Média dos Máximos: Comparação entre Controladores e Cenários')
    plt.xlabel('Geração')
    plt.ylabel('Fitness/Reward Máximo')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Gráfico para Simple GA - Apenas Médias
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers) * len(scenarios)))
    color_idx = 0
    
    for controller in controllers:
        for scenario in scenarios:
            simple_ga_avg_by_gen = {}
            
            # Coletar dados de todas as seeds
            for seed in results.keys():
                try:
                    simple_ga_data = results[seed][controller][scenario]['Simple_GA']
                    
                    # Agrupar por geração
                    for gen in sorted(simple_ga_data['Generation'].unique()):
                        gen_data = simple_ga_data[simple_ga_data['Generation'] == gen]
                        
                        # Média da população
                        if gen not in simple_ga_avg_by_gen:
                            simple_ga_avg_by_gen[gen] = []
                        simple_ga_avg_by_gen[gen].append(gen_data['Fitness'].mean())
                        
                except KeyError:
                    pass
            
            # Plotar Simple GA - Média da População
            if simple_ga_avg_by_gen:
                simple_ga_gens = sorted(simple_ga_avg_by_gen.keys())
                simple_ga_avg_means = [np.mean(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
                simple_ga_avg_stds = [np.std(simple_ga_avg_by_gen[gen]) for gen in simple_ga_gens]
                
                plt.plot(simple_ga_gens, simple_ga_avg_means, color=colors[color_idx], linewidth=2, 
                         marker='o', markersize=4, label=f'{controller}, {scenario}')
                plt.fill_between(simple_ga_gens, 
                                [m - s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                                [m + s for m, s in zip(simple_ga_avg_means, simple_ga_avg_stds)],
                                color=colors[color_idx], alpha=0.2)
                color_idx += 1
    
    plt.title('Simple GA - Média da População: Comparação entre Controladores e Cenários')
    plt.xlabel('Geração')
    plt.ylabel('Fitness Médio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Gráfico para Simple GA - Apenas Máximos
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers) * len(scenarios)))
    color_idx = 0
    
    for controller in controllers:
        for scenario in scenarios:
            simple_ga_max_by_gen = {}
            
       
            for seed in results.keys():
                try:
                    simple_ga_data = results[seed][controller][scenario]['Simple_GA']
                    
                    # Agrupar por geração
                    for gen in sorted(simple_ga_data['Generation'].unique()):
                        gen_data = simple_ga_data[simple_ga_data['Generation'] == gen]
                        
                        # Máximo da população
                        if gen not in simple_ga_max_by_gen:
                            simple_ga_max_by_gen[gen] = []
                        simple_ga_max_by_gen[gen].append(gen_data['Fitness'].max())
                        
                except KeyError:
                    pass
            
            # Plotar Simple GA - Média dos Máximos
            if simple_ga_max_by_gen:
                simple_ga_gens = sorted(simple_ga_max_by_gen.keys())
                simple_ga_max_means = [np.mean(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
                simple_ga_max_stds = [np.std(simple_ga_max_by_gen[gen]) for gen in simple_ga_gens]
                
                plt.plot(simple_ga_gens, simple_ga_max_means, color=colors[color_idx], linewidth=2, 
                         marker='s', markersize=4, label=f'{controller}, {scenario}')
                plt.fill_between(simple_ga_gens, 
                                [m - s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                                [m + s for m, s in zip(simple_ga_max_means, simple_ga_max_stds)],
                                color=colors[color_idx], alpha=0.2)
                color_idx += 1
    
    plt.title('Simple GA - Média dos Máximos: Comparação entre Controladores e Cenários')
    plt.xlabel('Geração')
    plt.ylabel('Fitness Máximo')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
 
    base_dir = os.getcwd()
    
    results = load_data(base_dir)

    controllers = set()
    scenarios = set()
    
    for seed in results:
        for controller in results[seed]:
            controllers.add(controller)
            for scenario in results[seed][controller]:
                scenarios.add(scenario)
    
    controllers = sorted(list(controllers))
    scenarios = sorted(list(scenarios))
    
    print(f"Controladores disponíveis: {controllers}")
    print(f"Cenários disponíveis: {scenarios}")
    
    print(results)
    
    run_comprehensive_analysis(results, controllers, scenarios)
    
    td.run_simplified_analysis(results)
    
    analyze_convergence_dual_combined(results, controllers, scenarios)
 
    plot_algorithm_specific_graphs(results, controllers, scenarios)
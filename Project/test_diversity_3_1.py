# ------------------- ANÁLISE DE DIVERSIDADE POPULACIONAL -------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ast import literal_eval

def parse_structure(structure_str):
    """
    Converte a string da estrutura em um array numpy 2D (5x5)
    """
    try:
        structure_list = literal_eval(structure_str)
        return np.array(structure_list).reshape(5, 5)
    except:
        print(f"Erro ao processar estrutura: {structure_str}")
        return None

def calculate_hamming_distance(struct1, struct2):
    """
    Calcula a distância de Hamming entre duas estruturas
    """
    if struct1 is None or struct2 is None:
        return np.nan
    
    diff_count = np.sum(struct1 != struct2)
    return diff_count / struct1.size

def calculate_diversity_metrics(data):
    """
    Calcula métricas de diversidade por geração para um conjunto de dados
    """
    diversity_by_gen = {}
    
    for gen in sorted(data['Generation'].unique()):
        gen_data = data[data['Generation'] == gen]
        
        if 'Structure' not in gen_data.columns or len(gen_data) < 2:
            continue
            
        structures = [parse_structure(s) for s in gen_data['Structure']]
        structures = [s for s in structures if s is not None]
        
        if len(structures) < 2:
            continue
            
        distances = []
        for i in range(len(structures)):
            for j in range(i+1, len(structures)):
                dist = calculate_hamming_distance(structures[i], structures[j])
                distances.append(dist)
        
        diversity_by_gen[gen] = {
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances)
        }
        
        # Calcular distribuição de tipos de voxels
        voxel_counts = np.zeros(5)  # Para os 5 tipos de voxels (0-4)
        
        for s in structures:
            for vtype in range(5):
                voxel_counts[vtype] += np.sum(s == vtype)
                
        total_voxels = sum(voxel_counts)
        voxel_props = voxel_counts / total_voxels
        
        for vtype in range(5):
            diversity_by_gen[gen][f'voxel_{vtype}_prop'] = voxel_props[vtype]
    
 
    if not diversity_by_gen:
        return pd.DataFrame()
        
    return pd.DataFrame.from_dict(diversity_by_gen, orient='index').reset_index().rename(columns={'index': 'Generation'})

def consolidated_diversity_analysis(results, metric_type='hamming'):
  
     # Cria gráficos consolidados das métricas de diversidade agrupadas por algoritmo
    


    alg_data = {
        'GA': {'data': [], 'label': []},
        'Simple_GA': {'data': [], 'label': []}
    }
    

    for seed in results:
        for controller in results[seed]:
            for scenario in results[seed][controller]:
                for alg_type in results[seed][controller][scenario]:
                    if alg_type not in alg_data:
                        continue
                        
                    data = results[seed][controller][scenario][alg_type]
                    
                    if 'Structure' not in data.columns:
                        continue
                        
                    # Calcular métricas de diversidade
                    diversity_metrics = calculate_diversity_metrics(data)
                    
                    if diversity_metrics.empty:
                        continue
                    
             
                    diversity_metrics['controller'] = controller
                    diversity_metrics['scenario'] = scenario
                    diversity_metrics['seed'] = seed
                    
                    alg_data[alg_type]['data'].append(diversity_metrics)
                    alg_data[alg_type]['label'].append(f"{controller}-{scenario}")
    

    fig_size = (14, 8)
    
    for alg_type in alg_data:
        if not alg_data[alg_type]['data']:
            continue
            
        # Combinar dados de diferentes controladores/cenários
        combined_data = pd.concat(alg_data[alg_type]['data'], ignore_index=True)
        
        if metric_type == 'hamming':
            # 1. Gráfico de diversidade de Hamming média por controlador e cenário
            plt.figure(figsize=fig_size)
            
            for controller in combined_data['controller'].unique():
                for scenario in combined_data['scenario'].unique():
                    subset = combined_data[(combined_data['controller'] == controller) & 
                                          (combined_data['scenario'] == scenario)]
                    
                    if not subset.empty:
                        plt.plot(subset['Generation'], subset['mean_distance'], 
                               label=f"{controller}-{scenario}", marker='o', markersize=4, alpha=0.8)
            
            plt.title(f'Diversidade de Hamming Média por Geração - {alg_type}')
            plt.xlabel('Geração')
            plt.ylabel('Distância de Hamming Média')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
            
            plt.show()
                
            # 2. Gráfico de comparação entre seeds para um controlador e cenário específico
            plt.figure(figsize=fig_size)
            
            ctrl_scen_counts = combined_data.groupby(['controller', 'scenario']).size()
            if not ctrl_scen_counts.empty:
                best_ctrl, best_scen = ctrl_scen_counts.idxmax()
                
                for seed in combined_data['seed'].unique():
                    subset = combined_data[(combined_data['controller'] == best_ctrl) & 
                                          (combined_data['scenario'] == best_scen) &
                                          (combined_data['seed'] == seed)]
                    
                    if not subset.empty:
                        plt.plot(subset['Generation'], subset['mean_distance'], 
                               label=f"Seed {seed}", marker='o', markersize=4)
                
                plt.title(f'Comparação de Seeds - {best_ctrl}-{best_scen} - {alg_type}')
                plt.xlabel('Geração')
                plt.ylabel('Distância de Hamming Média')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.show()
                
        elif metric_type == 'voxel':
            # 1. Gráfico de proporção média dos tipos de voxels por geração
            plt.figure(figsize=fig_size)
            
            # Selecionar um controlador e cenário como exemplo
            if not combined_data.empty:
                controllers = combined_data['controller'].unique()
                scenarios = combined_data['scenario'].unique()
                
                if len(controllers) > 0 and len(scenarios) > 0:
                    ctrl = controllers[0]
                    scen = scenarios[0]
                    
                    subset = combined_data[(combined_data['controller'] == ctrl) & 
                                          (combined_data['scenario'] == scen)]
                    
                    voxel_types = {
                        0: 'Vazio',
                        1: 'Rígido',
                        2: 'Macio',
                        3: 'At. Horizontal',
                        4: 'At. Vertical'
                    }
                    
                    for vtype in range(5):
                        plt.plot(subset['Generation'], subset[f'voxel_{vtype}_prop'], 
                               label=voxel_types[vtype], linewidth=2)
                    
                    plt.title(f'Proporção de Tipos de Voxels - {ctrl}-{scen} - {alg_type}')
                    plt.xlabel('Geração')
                    plt.ylabel('Proporção')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
                    plt.show()
                    
            # 2. Heatmap de distribuição de voxels por controlador/cenário na última geração
            plt.figure(figsize=fig_size)
            
            # Obter última geração para cada controlador/cenário
            last_gen_data = []
            
            for controller in combined_data['controller'].unique():
                for scenario in combined_data['scenario'].unique():
                    subset = combined_data[(combined_data['controller'] == controller) & 
                                          (combined_data['scenario'] == scenario)]
                    
                    if not subset.empty:
                        max_gen = subset['Generation'].max()
                        last_gen = subset[subset['Generation'] == max_gen].iloc[0]
                        
                        row_data = {
                            'controller_scenario': f"{controller}-{scenario}",
                            'Vazio': last_gen['voxel_0_prop'],
                            'Rígido': last_gen['voxel_1_prop'],
                            'Macio': last_gen['voxel_2_prop'],
                            'At. Horizontal': last_gen['voxel_3_prop'],
                            'At. Vertical': last_gen['voxel_4_prop']
                        }
                        
                        last_gen_data.append(row_data)
            
            if last_gen_data:
                df_last_gen = pd.DataFrame(last_gen_data)
                df_last_gen = df_last_gen.set_index('controller_scenario')
                
                sns.heatmap(df_last_gen, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': 'Proporção'})
                plt.title(f'Distribuição de Voxels na Última Geração - {alg_type}')
                plt.tight_layout()
    
                plt.show()
                
                
def run_simplified_analysis(results):
 
    consolidated_diversity_analysis(results, metric_type='hamming')
    
    consolidated_diversity_analysis(results, metric_type='voxel')
    
    print("Análise concluída!")

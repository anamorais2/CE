import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import glob
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# Set plotting style
plt.style.use('ggplot')

# 1. Load the summary data
def load_summary_data(file_path):
    # Handle the European/Brazilian format with comma as decimal separator
    df = pd.read_csv(file_path, sep=';')
    
    # Convert columns with comma to dot for decimal point
    for col in ['Best Fitness', 'Reward', 'Mutation Rate', 'Crossover Rate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    return df

# 2. Load per-generation data
def load_generation_data():
    all_data = []
    
    # Pattern to find all seed directories
    seed_dirs = glob.glob(os.path.join("results_seed_*"))
    
    print(f"Found {len(seed_dirs)} seed directories.")
    
    for seed_dir in seed_dirs:
        # Extract the seed number from the directory name
        seed_num = os.path.basename(seed_dir).split('_')[-1]
        
        # Find all CSV files in this seed directory
        for file_path in glob.glob(os.path.join(seed_dir, "**", "*.csv"), recursive=True):
            # Get the parent directory name which contains controller and scenario
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            # Parse the directory name to extract controller and scenario
            dir_parts = parent_dir.split('_')
            
            # Handle the naming pattern: [controller]_[scenario]
            if len(dir_parts) >= 2:
                # The last part is the scenario
                scenario = dir_parts[-1]
                # Everything before is the controller
                controller = '_'.join(dir_parts[:-1])
                
                # Extract the generation number from the filename
                file_basename = os.path.basename(file_path)
                generation = int(file_basename.split('_')[1].split('.')[0])
                
                try:
                    # Read the generation data
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                    
                    # Skip files with metadata headers (e.g., generation_0.csv)
                    if first_line.startswith("# ALGORITHM"):
                        print(f"Skipping metadata file: {file_path}")
                        continue
                    
                    gen_df = pd.read_csv(file_path)
                    gen_df['Scenario'] = scenario
                    gen_df['Controller'] = controller
                    gen_df['Generation'] = generation
                    gen_df['Seed'] = seed_num
                    all_data.append(gen_df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# 3. Check normality of data
def check_normality(data, column):
    """Test for normality and return appropriate test to use"""
    # Get unique scenario-controller combinations
    combinations = data.groupby(['Scenario', 'Controller'])
    
    results = {}
    for name, group in combinations:
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(group[column])
        
        # Store results
        results[name] = {
            'shapiro_stat': stat,
            'shapiro_p': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Create Q-Q plot
        plt.figure(figsize=(10, 6))
        qqplot(group[column], line='s', ax=plt.gca())
        plt.title(f'Q-Q Plot for {name[0]} - {name[1]} ({column})')
        plt.tight_layout()
        plt.savefig(f'qqplot_{name[0]}_{name[1]}_{column}.png')
        plt.close()
        
        print(f"Normality test for {name[0]} - {name[1]} ({column}):")
        print(f"  Shapiro-Wilk: statistic={stat:.4f}, p-value={p_value:.4f}")
        print(f"  Distribution is {'normal' if p_value > 0.05 else 'not normal'}")
    
    return results

# 4. Perform appropriate statistical test following the decision tree
def select_and_perform_test(data, column, scenario):
    """Select and perform the appropriate statistical test based on the decision tree"""
    scenario_data = data[data['Scenario'] == scenario]
    controllers = scenario_data['Controller'].unique()
    
    print(f"\n=== Statistical Analysis for {scenario} ({column}) ===")
    print(f"Result type: Continuous (1 variable)")
    print(f"Predictor type: Categorical (controllers)")
    print(f"Number of categories: {len(controllers)} ({'two' if len(controllers) == 2 else 'more than two'})")
    print(f"Measures: Different (different seeds)")
    
    # Check normality for each group
    normality_results = check_normality(scenario_data, column)
    all_normal = all(result['is_normal'] for result in normality_results.values())
    
    print(f"Parametric: {'Yes' if all_normal else 'No'} (based on Shapiro-Wilk test)")
    
    if len(controllers) > 2:
        if all_normal:
            # Perform Independent ANOVA
            groups = [scenario_data[scenario_data['Controller'] == ctrl][column].values 
                      for ctrl in controllers]
            f_stat, p_value = stats.f_oneway(*groups)
            
            print("\nSelected test: Independent ANOVA")
            print(f"Result: F={f_stat:.4f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("Conclusion: There are significant differences between controllers.")
                
                # Post-hoc: Tukey's HSD test
                tukey = pairwise_tukeyhsd(scenario_data[column], scenario_data['Controller'], alpha=0.05)
                print("\nPost-hoc test: Tukey HSD")
                print(tukey)
            else:
                print("Conclusion: There are no significant differences between controllers.")
        else:
            # Perform Kruskal-Wallis
            groups = [scenario_data[scenario_data['Controller'] == ctrl][column].values 
                      for ctrl in controllers]
            h_stat, p_value = stats.kruskal(*groups)
            
            print("\nSelected test: Kruskal-Wallis")
            print(f"Result: H={h_stat:.4f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("Conclusion: There are significant differences between controllers.")
                    
                # Post-hoc: Dunn's test with Bonferroni correction
                # Create a DataFrame for post-hoc analysis
                posthoc_data = pd.DataFrame({
                    'values': scenario_data[column],
                    'groups': scenario_data['Controller']
                })
                
                # Perform Dunn's test
                posthoc = sp.posthoc_dunn(posthoc_data, val_col='values', group_col='groups', p_adjust='bonferroni')
                
                print("\nPost-hoc test: Dunn's with Bonferroni correction")
                print("p-values:")
                print(posthoc)
                
                # Highlight significant differences
                print("\nSignificant differences (p < 0.05):")
                significant_pairs = []
                for i, ctrl1 in enumerate(controllers):
                    for j, ctrl2 in enumerate(controllers):
                        if i < j:  # To avoid duplicates
                            p = posthoc.loc[ctrl1, ctrl2]
                            if p < 0.05:
                                significant_pairs.append((ctrl1, ctrl2, p))
                
                if significant_pairs:
                    for ctrl1, ctrl2, p in significant_pairs:
                        print(f"- {ctrl1} vs {ctrl2}: p={p:.4f}")
   
            else:
                print("Conclusion: There are no significant differences between controllers.")
                
    elif len(controllers) == 2:
        # For two groups
        group1 = scenario_data[scenario_data['Controller'] == controllers[0]][column].values
        group2 = scenario_data[scenario_data['Controller'] == controllers[1]][column].values
        
        if all_normal:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            print("\nSelected test: Independent t-test")
            print(f"Result: t={t_stat:.4f}, p={p_value:.4f}")
        else:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group1, group2)
            
            print("\nSelected test: Mann-Whitney U")
            print(f"Result: U={u_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"Conclusion: There is a significant difference between {controllers[0]} and {controllers[1]}.")
        else:
            print(f"Conclusion: There is no significant difference between {controllers[0]} and {controllers[1]}.")

# 5. Create boxplots
def create_boxplots(data, column):
    """Create boxplots for the given column across all scenarios and controllers"""
    scenarios = data['Scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = data[data['Scenario'] == scenario]
        
        plt.figure(figsize=(12, 8))
        
        # Create boxplot
        ax = sns.boxplot(x='Controller', y=column, data=scenario_data)
        
        # Add individual points
        sns.stripplot(x='Controller', y=column, data=scenario_data, color='black', 
                     alpha=0.5, size=4, jitter=True)
        
        # Set title and labels
        if column == 'Best Fitness':
            title = f'Fitness Comparison ({scenario})'
            ylabel = 'Best Fitness'
        else:  # Reward
            title = f'Reward Comparison ({scenario})'
            ylabel = 'Reward Value'
            
        plt.title(title, fontsize=16, pad=20)
        plt.ylabel(ylabel, fontsize=14)
        plt.xlabel('Controller Type', fontsize=14)
        
        # Add mean values as text
        for i, controller in enumerate(scenario_data['Controller'].unique()):
            mean_val = scenario_data[scenario_data['Controller'] == controller][column].mean()
            ax.text(i, scenario_data[column].min() - 0.2, f'Mean: {mean_val:.2f}', 
                   horizontalalignment='center', size='medium', color='black', weight='semibold')
        
        plt.tight_layout()
        plt.savefig(f'boxplot_{scenario}_{column}.png', dpi=300)
        plt.close()

# 6. Create convergence plots from generation data
def create_convergence_plots(generation_data):
    """Create plots showing fitness/reward evolution over generations"""
    scenarios = generation_data['Scenario'].unique()
    controllers = generation_data['Controller'].unique()
    
    for scenario in scenarios:
        scenario_data = generation_data[generation_data['Scenario'] == scenario]
        
        # Plot fitness convergence
        plt.figure(figsize=(14, 8))
        
        for controller in controllers:
            controller_data = scenario_data[scenario_data['Controller'] == controller]
            
            # Group by generation and calculate mean fitness
            gen_means = controller_data.groupby('Generation')['Fitness'].mean()
            gen_std = controller_data.groupby('Generation')['Fitness'].std()
            
            # Plot with confidence interval
            generations = gen_means.index
            plt.plot(generations, gen_means.values, label=controller, linewidth=2)
            plt.fill_between(generations, 
                            gen_means.values - gen_std.values, 
                            gen_means.values + gen_std.values, 
                            alpha=0.2)
        
        plt.title(f'Fitness Convergence Over Generations ({scenario})', fontsize=16)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'fitness_convergence_{scenario}.png', dpi=300)
        plt.close()
        
        # Plot reward convergence if available
        if 'Reward' in generation_data.columns:
            plt.figure(figsize=(14, 8))
            
            for controller in controllers:
                controller_data = scenario_data[scenario_data['Controller'] == controller]
                
                # Group by generation and calculate mean reward
                gen_means = controller_data.groupby('Generation')['Reward'].mean()
                gen_std = controller_data.groupby('Generation')['Reward'].std()
                
                # Plot with confidence interval
                generations = gen_means.index
                plt.plot(generations, gen_means.values, label=controller, linewidth=2)
                plt.fill_between(generations, 
                                gen_means.values - gen_std.values, 
                                gen_means.values + gen_std.values, 
                                alpha=0.2)
            
            plt.title(f'Reward Convergence Over Generations ({scenario})', fontsize=16)
            plt.xlabel('Generation', fontsize=14)
            plt.ylabel('Reward', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'reward_convergence_{scenario}.png', dpi=300)
            plt.close()

# Main execution
if __name__ == "__main__":
    # Path to your summary CSV file
    summary_path = "final_results_summary.csv"
    
    # Path to directory containing per-generation data
    generation_data_path = "results_seed_*"
    
    # Load summary data
    summary_df = load_summary_data(summary_path)
    print(f"Summary data loaded: {len(summary_df)} rows")
    
    # Perform statistical analysis on summary data
    print("\n=== Statistical Analysis of Final Results ===")
    
    # Analyze each scenario separately
    for scenario in summary_df['Scenario'].unique():
        select_and_perform_test(summary_df, 'Best Fitness', scenario)
        select_and_perform_test(summary_df, 'Reward', scenario)
    
    # Create boxplots for final results
    create_boxplots(summary_df, 'Best Fitness')
    create_boxplots(summary_df, 'Reward')
    
    # Load and analyze per-generation data
    try:
        generation_df = load_generation_data()
        if not generation_df.empty:
            print(f"\nGeneration data loaded: {len(generation_df)} rows")
            create_convergence_plots(generation_df)
        else:
            print("\nNo generation data found or incorrect path")
    except Exception as e:
        print(f"\nError processing generation data: {e}")
    
    print("\nAnalysis complete! Files saved in the current directory.")
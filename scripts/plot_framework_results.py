import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_dir = "outputs/framework_fusion"
    output_dir = "outputs/framework_fusion/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gather all result CSV files
    csv_files = glob.glob(os.path.join(results_dir, "*_results.csv"))
    
    if not csv_files:
        print(f"No result CSV files found in {results_dir}")
        return

    # 2. Combine into a single DataFrame
    df_list = []
    for f in csv_files:
        # Extract metadata from filename if desired, though the csv already has 'model' and 'dataset'
        temp_df = pd.read_csv(f)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # If the backbone column doesn't exist from older runs, assume it was 'clip'
    if 'backbone' not in df.columns:
        df['backbone'] = 'clip'

    # Clean up method names for plotting
    df['fusion_method'] = df['model'].replace({
        'Torch_Early_Fusion_H0': 'Linear Probe',
        'Torch_Early_Fusion_H[128]': 'MLP Early Fusion',
        'Torch_Late_Fusion_H[128]': 'MLP Late Fusion'
    })
    
    # Create combined key for plotting
    df['model_architecture'] = df['backbone'].str.upper() + ' - ' + df['fusion_method']

    print("Combined Results DataFrame:")
    print(df[['dataset', 'backbone', 'fusion_method', 'accuracy', 'f1_score', 'auc']])

    # 3.5 Generate Summary Tables
    print("\nGenerating summary tables...")
    # Pivot the table to have datasets as rows and model architectural variants as columns 
    pivot_df = df.pivot(index='dataset', columns='model_architecture', values=['accuracy', 'f1_score', 'auc'])
    
    # Save as CSV
    table_csv_path = os.path.join(output_dir, "metrics_summary_table.csv")
    pivot_df.to_csv(table_csv_path)
    print(f"Saved summary CSV table to {table_csv_path}")
    
    # Save as Markdown (useful for Readme / GitHub)
    table_md_path = os.path.join(output_dir, "metrics_summary_table.md")
    with open(table_md_path, 'w') as f:
        f.write(pivot_df.to_markdown())
    print(f"Saved summary Markdown table to {table_md_path}")
    
    # Save as LaTeX (useful for papers)
    table_tex_path = os.path.join(output_dir, "metrics_summary_table.tex")
    with open(table_tex_path, 'w') as f:
        f.write(pivot_df.to_latex())
    print(f"Saved summary LaTeX table to {table_tex_path}\n")

    # 4. Plot Metrics
    metrics = {
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score',
        'auc': 'AUC'
    }

    sns.set_theme(style="whitegrid")
    
    for metric_col, metric_name in metrics.items():
        if metric_col not in df.columns or df[metric_col].isna().all():
            print(f"Skipping {metric_name} - completely missing or absent.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Use a barplot grouping by dataset and splitting by model architecture
        ax = sns.barplot(
            data=df,
            x='dataset',
            y=metric_col,
            hue='model_architecture',
            palette='viridis'
        )
        
        plt.title(f'{metric_name} Comparison by Backbone & Fusion Method', fontsize=14, pad=15)
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylim(0, 1.05) # Metrics are usually 0-1
        plt.xticks(rotation=45)
        plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save Plot
        plot_path = os.path.join(output_dir, f"{metric_col}_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot properly to {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
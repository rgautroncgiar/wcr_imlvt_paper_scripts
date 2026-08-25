import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pdb

plot_dir = Path('./plot')
plot_dir.mkdir(exist_ok=True)
xls_file_paths = {
    'rust': '../source_data/2025_WCR_master_results_Rust_Score_2025-08-27_.xlsx',
    'yield': '../source_data/2025_WCR_master_results_yield_2025-08-18_.xlsx',
}
common_index = True

genotypes_data_frames = {}
for data_type in ['yield', 'rust']:
    df = pd.read_excel(xls_file_paths[data_type],
                       sheet_name='varieties_scores_fa')
    df.columns = df.columns.str.lower()
    df = df.set_index('genotype')
    df.index = df.index.astype(str).str.strip()
    df = df.loc[df.index.str.casefold() != 'bp429a'].copy() # removing bp429a for rust
    df = df[['fa1', 'fa2']]
    df = df.round(2)
    genotypes_data_frames[data_type] = df

if common_index:
    all_genotypes = sorted(set().union(*[df.index for df in
                                         genotypes_data_frames.values()]))
    for data_type in genotypes_data_frames:
        genotypes_data_frames[data_type] = genotypes_data_frames[data_type].reindex(all_genotypes)

env_data_frames = {}
for data_type in ['yield', 'rust']:
    df = pd.read_excel(xls_file_paths[data_type], sheet_name='fa1_fa2_sites')
    df = df.set_index('site')[['fa1', 'fa2']]
    df = df.round(2)
    env_data_frames[data_type] = df

all_sites = sorted(set().union(*[df.index for df in
                                 env_data_frames.values()]))
for data_type in env_data_frames:
    env_data_frames[data_type] = env_data_frames[data_type].reindex(all_sites)

cell_height = 0.4
env_fig_height = len(all_sites) * cell_height
yticks = np.arange(len(all_sites)) + 0.5

for data_type in ['yield', 'rust']:
    df_heatmap = genotypes_data_frames[data_type]
    y_values = df_heatmap.values.flatten()
    y_values = y_values[~np.isnan(y_values)]
    vmax = np.abs(y_values).max()
    vmin = -vmax
    geno_fig_height = len(df_heatmap) * cell_height
    plt.figure(figsize=(5, geno_fig_height))
    ax = sns.heatmap(
        df_heatmap,
        cmap='PuOr',
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={"shrink": 0.4}
    )
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{data_type}_genotype_scores_heatmap.pdf',
                bbox_inches='tight')
    plt.close()

    env_df = env_data_frames[data_type]
    env_values = env_df.values.flatten()
    env_values = env_values[~np.isnan(env_values)]
    env_vmax = np.abs(env_values).max()
    env_vmin = -env_vmax
    plt.figure(figsize=(6, env_fig_height))
    ax = sns.heatmap(
        env_df,
        cmap='PuOr',
        vmin=env_vmin,
        vmax=env_vmax,
        center=0,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_kws={"shrink": 0.4}
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(all_sites, rotation=0)
    ax.set_ylim(len(all_sites), 0)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{data_type}_environment_fas_heatmap.pdf',
                bbox_inches='tight')
    plt.close()
print('Done')
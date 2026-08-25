from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import roman
from adjustText import adjust_text
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

CFG = {
    'scatter_figsize': (8, 6),
    'dendrogram_figsize': (7, 5),
    'point_mode': 'genetic_group',
    'point_size': 15,
    'point_size_default': 12,
    'point_edgecolor': 'black',
    'point_linewidth': 0.8,
    'radius_range': (2.0, 9.0),
    'invert_radius': {'yield': False, 'rust': True},
    'markers': ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h'],
    'palette': 'tab10',
    'genotype_label_size': 9,
    'genotype_label_alpha': 0.7,
    'site_label_size': 9,
    'label_push_pct': 0.1,
    'show_legend': True,
    'legend_fontsize': 8,
    'legend_loc': 2,
    'legend_marker_size': 6,
    'arrow_color': 'red',
    'arrow_alpha': 0.8,
    'arrow_head_width': 0.05,
    'arrow_head_length': 0.07,
    'arrow_scale_frac': 0.92,
    'arrow_percentile': 0,
    'label_percentile': 70,
    'initial_lim_pad': 1.05,
    'x_lim_pad': 1.10,
    'final_x_pad': 1.188,
    'final_y_pad': 1.08,
    'secondary_label_size': 8,
    'secondary_label_pos': 0.8,
    'secondary_label_pad': 14,
    'secondary_nbins': 6,
    'secondary_steps': [1, 2, 2.5, 5, 10],
    'secondary_min_ticks': 4,
    'secondary_powerlimits': (-3, 3),
    'site_connector_lw': 0.6,
    'genotype_connector_color': 'gray',
    'genotype_connector_lw': 0.5,
    'adjust_expand_points': (.5, .5),
    'adjust_expand_text': (1, 1),
    'adjust_force_points': (1, 1),
    'force_text': (.8,.8),
    'adjust_lim': 500,
    'axis_label_pos': 0.8,
    'axis_label_pad': 10,
    'zero_line_color': 'gray',
    'zero_line_lw': 0.5,
    'aspect': 'equal',
    'dendrogram_leaf_rotation': 90,
    'dendrogram_leaf_font_size': 8,
    'dendrogram_above_color': '#AAAAAA',
    'dendrogram_group_bottom': 0.06,
    'dendrogram_group_y': 0.013,
    'dendrogram_group_size': 11,
    'dendrogram_group_weight': 'bold',
    'save_bbox': 'tight',
    'secondary_x_position': 'top',
    'secondary_y_position': 'right',
    'site_x_label': 'rot. site fa1 loadings',
    'site_y_label': 'rot. site fa2 loadings',
    'genotype_x_label': 'rot. genotype fa1 score',
    'genotype_y_label': 'rot. genotype fa2 score',
    'dendrogram_y_label': 'Complete linkage distance',
    'genetic_legend_title': 'genetic group',
    'cluster_legend_title': 'cluster',
    'annotate_ticks': False,
    'label_groups': True,
    'agg_method': 'complete',
    'k_map': {'yield': 4, 'rust': 3},
    'axis_info': {
        'yield': {'fa1': 35.7, 'fa2': 21.9},
        'rust': {'fa1': 62.3, 'fa2': 5.0},
    },
}

XLS = {
    'rust': '../source_data/2025_WCR_master_results_Rust_Score_2025-08-27_.xlsx',
    'yield': '../source_data/2025_WCR_master_results_yield_2025-08-18_.xlsx',
}


def k_color_threshold(Z, k):
    n = Z.shape[0] + 1
    d = Z[:, 2]
    if k <= 1:
        return d[-1] + 1e-6
    if k >= n:
        return d[0] - 1e-6
    i = n - k - 1
    return 0.5 * (d[i] + d[i + 1])


def agglom_on_diff(fa1, fa2, k, method='ward'):
    Z = linkage(np.column_stack([fa1, fa2]), method=method)
    return fcluster(Z, t=k, criterion='maxclust'), Z


def _idealdist_sizes(fa1, fa2, radius_range=(2.0, 9.0), invert=True, eps=1e-12):
    fa1, fa2 = np.asarray(fa1, float), np.asarray(fa2, float)
    fa1_ideal = np.min(fa1) if invert else np.max(fa1)
    measure = np.sqrt((fa1 - fa1_ideal) ** 2 + fa2 ** 2)
    measure = measure.max() - measure
    m_min, m_max = measure.min(), measure.max()
    if m_max - m_min < eps:
        return np.full_like(measure, np.mean(radius_range)) ** 2
    r = radius_range[0] + np.ptp(radius_range) * (measure - m_min) / (m_max - m_min)
    return r ** 2


def load_variety_description():
    df = pd.read_excel('../source_data/VarietyDescription.xlsx', sheet_name=0)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    for col in df.select_dtypes('object'):
        df[col] = df[col].str.lower().str.strip().str.replace(' ', '_')
    df = df.rename(columns={'variety_name': 'genotype'})
    df['main_category'] = df['main_category'].replace({'hybrid': 'F1_hybrid'}).str.replace('_', ' ')
    return df


def plot_scatter(df_genotypes, df_sites, cluster_to_color, out_path, point_mode='cluster', invert_radius=True,
                 axis_info=None, group_names=None, cluster_to_marker=None, grouping_col='cluster'):
    fig, ax = plt.subplots(figsize=CFG['scatter_figsize'])
    if point_mode == 'idealdist':
        sizes = _idealdist_sizes(df_genotypes['fa1'], df_genotypes['fa2'], CFG['radius_range'], invert_radius)
        ax.scatter(df_genotypes['fa1'], df_genotypes['fa2'], s=sizes, facecolors='none',
                   edgecolors=CFG['point_edgecolor'], linewidths=CFG['point_linewidth'])
    else:
        if cluster_to_marker is not None:
            for c in np.unique(df_genotypes[grouping_col]):
                m = df_genotypes[grouping_col] == c
                ax.scatter(df_genotypes.loc[m, 'fa1'], df_genotypes.loc[m, 'fa2'], c=cluster_to_color[c],
                           s=CFG['point_size'], marker=cluster_to_marker[c])
        else:
            ax.scatter(df_genotypes['fa1'], df_genotypes['fa2'],
                       c=[cluster_to_color[c] for c in df_genotypes[grouping_col]], s=CFG['point_size_default'])
        if CFG['show_legend']:
            name = lambda c: group_names[c] if group_names is not None and c in group_names else str(c)
            handles = [plt.Line2D([0], [0], marker=cluster_to_marker[c] if cluster_to_marker else 'o', linestyle='',
                                  markersize=CFG['legend_marker_size'], markerfacecolor=cluster_to_color[c],
                                  markeredgecolor=cluster_to_color[c])
                       for c in sorted(cluster_to_color, key=name)]
            labels = [name(c).upper() for c in sorted(cluster_to_color, key=name)]
            title = CFG['genetic_legend_title'] if grouping_col == 'main_category' else CFG['cluster_legend_title']
            ax.legend(handles, labels, frameon=False, fontsize=CFG['legend_fontsize'], loc=CFG['legend_loc'], title=title)

    genotype_texts = [ax.text(r.fa1, r.fa2, r.variety, fontsize=CFG['genotype_label_size'],
                              alpha=CFG['genotype_label_alpha']) for r in df_genotypes.itertuples()]
    gx, gy = df_genotypes['fa1'].to_numpy(), df_genotypes['fa2'].to_numpy()
    max_x = np.nanmax(np.abs(gx)) if gx.size else 1.0
    max_y = np.nanmax(np.abs(gy)) if gy.size else 1.0
    x_lim = CFG['initial_lim_pad'] * (max_x if max_x > 0 else 1.0) * CFG['x_lim_pad']
    y_lim = CFG['initial_lim_pad'] * (max_y if max_y > 0 else 1.0)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    df_sites = df_sites.copy()
    df_sites['norm'] = df_sites['fa1'] ** 2 + df_sites['fa2'] ** 2
    smax_x = float(np.nanmax(np.abs(df_sites['fa1']))) if len(df_sites) else 1
    smax_y = float(np.nanmax(np.abs(df_sites['fa2']))) if len(df_sites) else 1
    cand_x = x_lim / smax_x if smax_x > 0 else np.inf
    cand_y = y_lim / smax_y if smax_y > 0 else np.inf
    arrow_scale = CFG['arrow_scale_frac'] * min(cand_x, cand_y) if np.isfinite(min(cand_x, cand_y)) else 1
    arrow_thresh = np.percentile(df_sites['norm'], CFG['arrow_percentile'])
    label_thresh = np.percentile(df_sites['norm'], CFG['label_percentile'])

    site_texts = []
    for r in df_sites.itertuples():
        if r.norm >= arrow_thresh:
            ax.arrow(0, 0, r.fa1 * arrow_scale, r.fa2 * arrow_scale,
                     head_width=CFG['arrow_head_width'], head_length=CFG['arrow_head_length'],
                     fc=CFG['arrow_color'], ec=CFG['arrow_color'], alpha=CFG['arrow_alpha'], length_includes_head=True)
        if r.norm >= label_thresh:
            f = arrow_scale * (1 + CFG['label_push_pct'])
            site_texts.append(ax.text(r.fa1 * f, r.fa2 * f, r.site,
                                      color=CFG['arrow_color'], fontsize=CFG['site_label_size']))

    if len(df_sites):
        f = arrow_scale * (1 + CFG['label_push_pct'])
        sx, sy = (df_sites['fa1'] * f).to_numpy(), (df_sites['fa2'] * f).to_numpy()
        ax.set_xlim(-CFG['final_x_pad'] * max(x_lim, np.nanmax(np.abs(sx))),
                    CFG['final_x_pad'] * max(x_lim, np.nanmax(np.abs(sx))))
        ax.set_ylim(-CFG['final_y_pad'] * max(y_lim, np.nanmax(np.abs(sy))),
                    CFG['final_y_pad'] * max(y_lim, np.nanmax(np.abs(sy))))

    if arrow_scale not in (0, 1, np.inf) and np.isfinite(arrow_scale):
        secax_x = ax.secondary_xaxis(CFG['secondary_x_position'], functions=(lambda x: x / arrow_scale, lambda u: u * arrow_scale))
        secax_y = ax.secondary_yaxis(CFG['secondary_y_position'], functions=(lambda y: y / arrow_scale, lambda v: v * arrow_scale))
        secax_x.set_xlabel(CFG['site_x_label'], x=CFG['secondary_label_pos'], ha='right', labelpad=CFG['secondary_label_pad'])
        secax_y.set_ylabel(CFG['site_y_label'], y=CFG['secondary_label_pos'], va='top', rotation=90,
                           labelpad=CFG['secondary_label_pad'])
        for secax, axis in ((secax_x, 'x'), (secax_y, 'y')):
            secax.tick_params(axis=axis, colors=CFG['arrow_color'], labelsize=CFG['secondary_label_size'])
            secax.xaxis.label.set_color(CFG['arrow_color']) if axis == 'x' else secax.yaxis.label.set_color(CFG['arrow_color'])
        locator = MaxNLocator(nbins=CFG['secondary_nbins'], steps=CFG['secondary_steps'], symmetric=True,
                             min_n_ticks=CFG['secondary_min_ticks'])
        secax_x.xaxis.set_major_locator(locator)
        secax_y.yaxis.set_major_locator(locator)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits(CFG['secondary_powerlimits'])
        fmt.set_useOffset(False)
        secax_x.xaxis.set_major_formatter(fmt)
        secax_y.yaxis.set_major_formatter(fmt)

    adjust_text(site_texts, ax=ax,
                arrowprops={'arrowstyle': '-', 'color': CFG['arrow_color'], 'lw': CFG['site_connector_lw']})
    adjust_text(genotype_texts, gx, gy, ax=ax,
                arrowprops={'arrowstyle': '-', 'color': CFG['genotype_connector_color'], 'lw': CFG['genotype_connector_lw']},
                expand_points=CFG['adjust_expand_points'], expand_text=CFG['adjust_expand_text'],
                force_points=CFG['adjust_force_points'], only_move={'text': 'xy'}, lim=CFG['adjust_lim'],
                force_text=CFG['force_text'])
    ax.set_xlabel(f"{CFG['genotype_x_label']} ({axis_info['fa1']}%)", ha='right', x=CFG['axis_label_pos'], labelpad=CFG['axis_label_pad'])
    ax.set_ylabel(f"{CFG['genotype_y_label']} ({axis_info['fa2']}%)", va='top', rotation=90,
                  y=CFG['axis_label_pos'], labelpad=CFG['axis_label_pad'])
    ax.axhline(0, color=CFG['zero_line_color'], lw=CFG['zero_line_lw'])
    ax.axvline(0, color=CFG['zero_line_color'], lw=CFG['zero_line_lw'])
    ax.set_aspect(CFG['aspect'], adjustable='box')
    fig.savefig(out_path, bbox_inches=CFG['save_bbox'])
    plt.close(fig)


def plot_dendrogram(Z, df_genotypes, cluster_to_color, k, out_path, group_names=None, grouping_col='cluster'):
    t = k_color_threshold(Z, k)
    n = Z.shape[0] + 1
    children = {i + n: (int(Z[i, 0]), int(Z[i, 1])) for i in range(Z.shape[0])}
    memo = {}

    def leaves(i):
        if i < n:
            return [i]
        if i not in memo:
            a, b = children[i]
            memo[i] = leaves(a) + leaves(b)
        return memo[i]

    y = df_genotypes[grouping_col].to_numpy()

    def link_color_func(i):
        s = {y[j] for j in leaves(i)}
        return cluster_to_color[next(iter(s))] if len(s) == 1 else CFG['dendrogram_above_color']

    fig, ax = plt.subplots(figsize=CFG['dendrogram_figsize'])
    d = dendrogram(Z, labels=df_genotypes['variety'].astype(str).tolist(), color_threshold=t,
                   leaf_rotation=CFG['dendrogram_leaf_rotation'], leaf_font_size=CFG['dendrogram_leaf_font_size'],
                   above_threshold_color=CFG['dendrogram_above_color'], link_color_func=link_color_func, ax=ax)
    cr = y[d['leaves']]
    clusters_sorted = df_genotypes.groupby(grouping_col)['fa1'].mean().sort_values(ascending=False).index.tolist()
    if group_names is None:
        group_names = {c: string.ascii_uppercase[j] for j, c in enumerate(clusters_sorted)}
    for i, lbl in enumerate(ax.get_xmajorticklabels()):
        lbl.set_color(cluster_to_color[cr[i]])
        if CFG['annotate_ticks']:
            lbl.set_text(f"{group_names[cr[i]]} | {lbl.get_text()}")
    if CFG['label_groups']:
        pos = ax.get_xticks()
        ymax = ax.get_ylim()[1]
        ax.set_ylim(-CFG['dendrogram_group_bottom'] * ymax, ymax)
        i0 = 0
        for i in range(1, len(cr) + 1):
            if i == len(cr) or cr[i] != cr[i0]:
                xm = 0.5 * (pos[i0] + pos[i - 1])
                ax.text(xm, -CFG['dendrogram_group_y'] * ymax, group_names[cr[i0]], ha='center', va='top',
                        fontsize=CFG['dendrogram_group_size'], color=cluster_to_color[cr[i0]],
                        weight=CFG['dendrogram_group_weight'])
                i0 = i
    ax.set_ylabel(CFG['dendrogram_y_label'])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches=CFG['save_bbox'])
    plt.close(fig)
    return group_names


def main():
    csv_dir, plot_dir = Path('./csv'), Path('./plot')
    csv_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    tab10 = [to_hex(c) for c in mpl.colormaps[CFG['palette']].colors]
    df_variety = load_variety_description() if CFG['point_mode'] == 'genetic_group' else None

    if df_variety is not None:
        categories = sorted(df_variety['main_category'].dropna().unique())
        palette = (tab10 * (len(categories) // len(tab10) + 1))[:len(categories)]
        global_colors = dict(zip(categories, palette))
        global_markers = {c: CFG['markers'][i % len(CFG['markers'])] for i, c in enumerate(categories)}
        global_names = {c: c for c in categories}
    else:
        global_colors = global_markers = global_names = None

    for data_type in ['yield', 'rust']:
        k = CFG['k_map'][data_type]
        df_sites = pd.read_excel(XLS[data_type], sheet_name='fa1_fa2_sites')
        df_sites.columns = df_sites.columns.str.lower()
        df_sites = df_sites.dropna(subset=['site', 'blue', 'fa1', 'fa2'])
        df_genotypes = pd.read_excel(XLS[data_type], sheet_name='varieties_scores_fa')
        df_genotypes.columns = df_genotypes.columns.str.lower()
        df_genotypes = df_genotypes.rename(columns={'genotype': 'variety'}).dropna(subset=['variety', 'fa1', 'fa2'])
        mask = df_genotypes['variety'].astype(str).str.strip().str.casefold().isin({'bp429a'})
        df_genotypes = df_genotypes.loc[~mask].copy()
        df_genotypes['variety_lower'] = df_genotypes['variety'].str.lower()

        if CFG['point_mode'] == 'genetic_group' and df_variety is not None:
            df_genotypes = df_genotypes.merge(df_variety[['genotype', 'main_category']], left_on='variety_lower',
                                              right_on='genotype', how='left').drop(columns=['genotype', 'variety_lower'])
            df_genotypes = df_genotypes.dropna(subset=['main_category'])
            grouping_col = 'main_category'
            cluster_to_color, cluster_to_marker, group_names = global_colors, global_markers, global_names
        else:
            df_genotypes = df_genotypes.drop(columns=['variety_lower'])
            grouping_col = 'cluster'

        df_genotypes['variety'] = df_genotypes['variety'].str.replace('Kartila1', 'Kartika1')
        X = df_genotypes[['fa1', 'fa2']].to_numpy()
        y, Z = agglom_on_diff(X[:, 0], X[:, 1], k, CFG['agg_method'])
        df_genotypes['cluster'] = y

        if grouping_col == 'cluster':
            clusters = df_genotypes.groupby('cluster')['fa1'].mean().sort_values(ascending=False).index.tolist()
            group_names = {c: roman.toRoman(i + 1) if data_type == 'rust' else string.ascii_uppercase[i]
                           for i, c in enumerate(clusters)}
            palette = (tab10 * (len(clusters) // len(tab10) + 1))[:len(clusters)]
            cluster_to_color = dict(zip(clusters, palette))
            cluster_to_marker = {c: CFG['markers'][i % len(CFG['markers'])] for i, c in enumerate(clusters)}

        suffix = f"_{CFG['point_mode']}" if CFG['point_mode'] == 'genetic_group' else ''
        df_genotypes[['variety', 'fa1', 'fa2', grouping_col]].to_csv(csv_dir / f'clusters_{data_type}{suffix}.csv', index=False)
        plot_dendrogram(Z, df_genotypes, cluster_to_color, k, plot_dir / f'fa_dendrogram_{data_type}{suffix}.pdf',
                        group_names=group_names, grouping_col=grouping_col)
        plot_scatter(df_genotypes, df_sites, cluster_to_color, plot_dir / f'fa_biplot_{data_type}{suffix}.pdf',
                     point_mode=CFG['point_mode'], invert_radius=CFG['invert_radius'].get(data_type, True),
                     axis_info=CFG['axis_info'][data_type], group_names=group_names,
                     cluster_to_marker=cluster_to_marker, grouping_col=grouping_col)

    print('Biplots, dendrograms, and clusters saved.')


if __name__ == '__main__':
    main()

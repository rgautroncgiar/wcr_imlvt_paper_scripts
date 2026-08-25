from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
from pathlib import Path
import pdb
import csv
from pprint import pprint

np.random.seed(123)

def build_func_tree(d1, d2, func):
    result = {}
    for key in d1:
        v1 = d1[key]
        v2 = d2[key]
        if isinstance(v1, dict) and isinstance(v2, dict):
            result[key] = build_func_tree(v1, v2, func)
        elif not isinstance(v1, dict) and not isinstance(v2, dict):
            result[key] = func(v1, v2)
        else:
            raise TypeError(
                f"Type structure mismatch at key {key!r}: "
                f"{type(v1).__name__} vs {type(v2).__name__}"
            )
    return result

def combine_dicts(list_of_dicts):
    if not list_of_dicts:
        return {}
    result = {}
    for key in list_of_dicts[0].keys():
        values = [d[key] for d in list_of_dicts]
        if isinstance(values[0], dict):
            result[key] = combine_dicts(values)
        else:
            result[key] = values
    return result

def write_pvalues_to_csv(data, filename, header=None, key_order=None):
    default_header = ['factor', 'outcome', 'term', 'p_value']
    header = header or default_header
    factors = list(data.keys())
    if key_order and 'factor' in key_order:
        factors = [f for f in key_order['factor'] if f in data]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for factor in factors:
            outcomes = data.get(factor, {})
            outcome_keys = list(outcomes.keys())
            if key_order and 'outcome' in key_order:
                outcome_keys = [o for o in key_order['outcome'] if o in outcomes]
            for outcome in outcome_keys:
                terms = outcomes.get(outcome, {})
                term_keys = list(terms.keys())
                if key_order and 'term' in key_order:
                    term_keys = [t for t in key_order['term'] if t in terms]
                for term in term_keys:
                    p = terms.get(term, None)
                    row_data = {
                        'factor': factor,
                        'outcome': outcome,
                        'term': term,
                        'p_value': float(p) if p is not None else ''
                    }
                    row = [row_data.get(col, '') for col in header]
                    writer.writerow(row)

def get_p_value(observation, distribution):
    observation = np.abs(observation)
    distribution = np.asarray(distribution)
    return (distribution >= observation).mean()

def compute_explained_variance(pls, y):
    t_scores = pls.x_scores_
    p_loadings = pls.x_loadings_
    explained_variances = {'y': {}, 'X': {}}
    t1, t2 = t_scores.T
    p1, p2 = p_loadings.T
    N, _ = t_scores.shape
    M, _ = p_loadings.shape
    corr_y_t1 = np.corrcoef(y, t1)[0,1]
    corr_y_t2 = np.corrcoef(y, t2)[0,1]
    explained_variances['y']['t1'] = corr_y_t1 ** 2
    explained_variances['y']['t2'] = corr_y_t2 ** 2
    explained_variances['y']['t1,t2'] = explained_variances['y']['t1'] + explained_variances['y']['t2']
    explained_variances['X']['t1'] = (t1**2).sum()*(p1**2).sum()/(N-1)/M
    explained_variances['X']['t2'] = (t2**2).sum()*(p2**2).sum()/(N-1)/M
    explained_variances['X']['t1,t2'] = explained_variances['X']['t1'] + explained_variances['X']['t2']
    return explained_variances

def save_correlations_csv(
    pls_model, X_data, y_values, target_name, filename,
    supplementary_vars=None, supplementary_data=None
):
    T = pls_model.x_scores_[:, :2]
    rows = []
    for col in X_data.columns:
        rows.append({
            'variable': col,
            'type': 'bioclimatic',
            'corr_t1': np.corrcoef(X_data[col], T[:, 0])[0, 1],
            'corr_t2': np.corrcoef(X_data[col], T[:, 1])[0, 1],
        })
    if supplementary_vars is not None and supplementary_data is not None:
        for var in supplementary_vars:
            if var in supplementary_data:
                rows.append({
                    'variable': var,
                    'type': 'supplementary',
                    'corr_t1': np.corrcoef(supplementary_data[var], T[:, 0])[0, 1],
                    'corr_t2': np.corrcoef(supplementary_data[var], T[:, 1])[0, 1],
                })
    rows.append({
        'variable': target_name,
        'type': 'target',
        'corr_t1': np.corrcoef(y_values, T[:, 0])[0, 1],
        'corr_t2': np.corrcoef(y_values, T[:, 1])[0, 1],
    })
    pd.DataFrame(rows).to_csv(filename, index=False, float_format='%.4f')

def add_geo_features(df, lat_col='latitude', lon_col='longitude', greenwitch_ref=False):
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Expected columns '{lat_col}' and '{lon_col}'.")
    out = df.copy()
    phi = np.radians(out[lat_col].astype(float))
    lam = np.radians(out[lon_col].astype(float))
    out['northwardness']   = np.sin(phi)
    out['equatoriality'] = np.cos(phi)
    if greenwitch_ref:
        lam0 = 0
    else:
        lam0 = np.arctan2(np.sin(lam).mean(), np.cos(lam).mean())
    out['east_west_rel'] = np.sin(lam - lam0)
    out['meridian_rel']  = np.cos(lam - lam0)
    return out, ['northwardness', 'equatoriality', 'east_west_rel', 'meridian_rel']

def dict_to_csv(dict, filename):
    coeff_df = pd.DataFrame(dict).T
    coeff_df.index.name = "factor"
    coeff_df.to_csv(filename, index=True, float_format='%.3f')

def plot_correlation_circle(pls_model, feature_names, target_name, explained_var_dict, rownames=None, y_values=None, supplementary_vars=None, supplementary_data=None, X_data=None, color_data=None, color_label=None):
    if X_data is None:
        raise ValueError("X_data is required")
    if y_values is None:
        raise ValueError("y_values is required")
    T = pls_model.x_scores_[:, :2]
    P_corr = np.zeros((len(feature_names), 2))
    for i, feat in enumerate(feature_names):
        P_corr[i, 0] = np.corrcoef(X_data[feat], T[:, 0])[0, 1]
        P_corr[i, 1] = np.corrcoef(X_data[feat], T[:, 1])[0, 1]
    supplementary_corrs = None
    if supplementary_vars is not None and supplementary_data is not None:
        supplementary_corrs = np.zeros((len(supplementary_vars), 2))
        for i, var in enumerate(supplementary_vars):
            if var in supplementary_data.columns:
                supplementary_corrs[i, 0] = np.corrcoef(supplementary_data[var], T[:, 0])[0, 1]
                supplementary_corrs[i, 1] = np.corrcoef(supplementary_data[var], T[:, 1])[0, 1]
    r_y = np.array([np.corrcoef(y_values, T[:, 0])[0, 1], np.corrcoef(y_values, T[:, 1])[0, 1]])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    blue_color = "#0059FF"
    red_color = "#FF0000"
    supp_color = "#0059FF"
    dark_blue = "#001329"
    dark_red = "#E60101"
    dark_supp = "#001329"
    label_distance = 0.04
    circle_xlabel = 'correlation with t1'
    circle_ylabel = 'correlation with t2'
    scatter_xlabel = f"t1 (exp. var. X: {explained_var_dict['X']['t1']:.1%}, y ({target_name}):{explained_var_dict['y']['t1']:.1%})"
    scatter_ylabel = f"t2 (exp. var. X: {explained_var_dict['X']['t2']:.1%}, y ({target_name}):{explained_var_dict['y']['t2']:.1%})"
    for ax, axes_labels in zip((ax0, ax1), ((circle_xlabel, circle_ylabel),(scatter_xlabel,scatter_ylabel))):
        ax.set_aspect("equal", "box")
        ax.axhline(y=0, color='k', lw=0.5)
        ax.axvline(x=0, color='k', lw=0.5)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
    ax0.set_xticks(np.arange(-1, 1.01, 0.2))
    ax0.set_yticks(np.arange(-1, 1.01, 0.2))
    ax0.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    ax0.add_patch(plt.Circle((0, 0), 1, fc="none", ec="0", ls="-"))
    ax0.add_patch(plt.Circle((0, 0), 0.5, fc="none", ec="0", ls=(0, (5,10)), lw=0.8))
    texts = []
    text_targets = []
    for (x, y), n in zip(P_corr, feature_names):
        ax0.arrow(0, 0, x, y, width=0.002, head_width=0.04, length_includes_head=True, color=blue_color)
        dist = np.hypot(x, y)
        if dist == 0:
            label_x, label_y = x, y
        else:
            label_x = x + label_distance * x / dist
            label_y = y + label_distance * y / dist
        texts.append(ax0.text(
            label_x, label_y, n, fontsize=9, color=dark_blue, zorder=5,
            ha='center', va='center'
        ))
        text_targets.append((x, y))
    if supplementary_vars is not None and supplementary_corrs is not None:
        for (x, y), n in zip(supplementary_corrs, supplementary_vars):
            if not (np.isnan(x) or np.isnan(y)):
                ax0.arrow(0, 0, x, y, width=0.002, head_width=0.04, length_includes_head=True, color=supp_color, linestyle=(0, (3, 5)), alpha=0.8, fill=False)
                dist = np.hypot(x, y)
                if dist == 0:
                    label_x, label_y = x, y
                else:
                    label_x = x + label_distance * x / dist
                    label_y = y + label_distance * y / dist
                texts.append(ax0.text(
                    label_x, label_y, n, fontsize=9,
                    color=dark_supp, zorder=5, ha='center',
                    va='center', style='italic'
                ))
                text_targets.append((x, y))
    ax0.arrow(0, 0, r_y[0], r_y[1], width=0.003, head_width=0.06, length_includes_head=True, color=red_color)
    dist = np.hypot(r_y[0], r_y[1])
    if dist == 0:
        label_x, label_y = r_y[0], r_y[1]
    else:
        label_x = r_y[0] + label_distance * r_y[0] / dist
        label_y = r_y[1] + label_distance * r_y[1] / dist
    texts.append(ax0.text(
        label_x, label_y, target_name, fontsize=10, weight="bold",
        color=red_color, zorder=5, ha='center', va='center'
    ))
    text_targets.append((r_y[0], r_y[1]))
    target_x, target_y = zip(*text_targets)
    adjust_text(
        texts,
        ax=ax0,
        target_x=target_x,
        target_y=target_y,
        x=target_x,
        y=target_y,
        force_points=(0.8, 0.8),
        force_text=(0.4, 0.4),
        expand_points=(1.05, 1.05),
        expand_text=(1.05, 1.05),
        arrowprops=dict(
            arrowstyle="-",
            color='gray',
            lw=0.7,
            alpha=0.6,
        ),
        autoalign='xy',
        ensure_inside_axes=True,
        add_lines=True,
        lim=500,
        shrinkA=5,
        shrinkB=0,
    )
    if rownames is not None:
        c_values = y_values if color_data is None else color_data
        c_label = target_name if color_label is None else color_label
        if c_values is not None:
            c_values_numeric = pd.to_numeric(c_values, errors='coerce')
            if pd.api.types.is_numeric_dtype(c_values_numeric) and not c_values_numeric.isnull().all():
                max_abs = np.nanmax(np.abs(c_values_numeric))
                min_val = np.nanmin(c_values_numeric)

                if c_label in ['fa1', 'fa2']:
                    cmap = 'PuOr'
                    vmin = -max_abs
                    vmax = max_abs
                else:
                    cmap = 'viridis' if min_val >= 0 else 'PuOr'
                    vmin = -max_abs if min_val < 0 else min_val
                    vmax = max_abs

                scatter = ax1.scatter(
                    T[:, 0], T[:, 1],
                    s=80,
                    c=c_values_numeric,
                    cmap=cmap,
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=1.2,
                    vmin=vmin,
                    vmax=vmax
                )

                cbar = fig.colorbar(scatter, ax=ax1, shrink=0.8)
                cbar.set_label(c_label, rotation=270, labelpad=15)
            else:
                 ax1.scatter(T[:, 0], T[:, 1], s=20, marker='.', color='k', alpha=0.8)
        else:
            ax1.scatter(T[:, 0], T[:, 1], s=20, marker='.', color='k', alpha=0.8)
        texts = []
        for (x, y), r in zip(T, rownames):
            texts.append(ax1.text(x, y, r, fontsize=7, color='k', zorder=5, fontweight='bold'))
        adjust_text(texts, ax=ax1, force_points=(0.6, 0.6), force_text=(0.4, 0.4), expand_points=(1.05, 1.05), expand_text=(1.05, 1.05), arrowprops=dict(arrowstyle="-", color='gray', lw=0.7, alpha=0.6), autoalign='xy', ensure_inside_axes=True, add_lines=True, lim=500, shrinkA=5, shrinkB=5)
    return fig, (ax0, ax1), supplementary_corrs

def dict_to_metric_df(data, metric_name):
    rows = [
        {'factor': factor, 'outcome': outcome, 'term': term, metric_name: float(val)}
        for factor, outcomes in data.items()
        for outcome, terms in outcomes.items()
        for term, val in terms.items()
    ]
    df = pd.DataFrame(rows).sort_values(by=['outcome', 'factor'])[['outcome', 'factor','term', metric_name]]
    return df

def get_pls_reg_coeffs(pls_model, X, scaled=True):
    intercept = pls_model.intercept_
    if scaled:
        coeffs = np.dot(pls_model.x_rotations_, pls_model.y_loadings_.T).flatten()
        y_std_scalar = np.asarray(pls_model._y_std).item()
        intercept /= y_std_scalar
    else:
        coeffs = pls_model.coef_[0]
    results = {'intercept': np.asarray(intercept).item()}
    for coeff, x_name in zip(coeffs, X.columns.values):
        results[x_name] = coeff
    return results

def plot_pred_vs_true(y_true, y_pred, target_name, rownames=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=1, color='k', s=4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', lw=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    ax.set_xlabel(f'true {target_name}')
    ax.set_ylabel(f'predicted {target_name}')
    if rownames is not None:
        texts = []
        for (x, y), r in zip(zip(y_true, y_pred), rownames):
            texts.append(ax.text(x, y, r, fontsize=8, color='k', zorder=5))
        adjust_text(texts, ax=ax, force_points=(0.8, 0.8), force_text=(0.4, 0.4), expand_points=(1.05, 1.05), expand_text=(1.05, 1.05), arrowprops=dict(arrowstyle="-", color='gray', lw=0.7, alpha=0.6), autoalign='xy', ensure_inside_axes=True, add_lines=True, lim=500, shrinkA=5, shrinkB=5)
    return fig, ax

def print_dict_by_abs(d, reverse=False):
    for k, v in sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=reverse):
        print(f"{k}: {v}")

def get_vip(model, X):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vip_calc = np.asarray(p*(s.T @ weight)/total_s).item()
        vips[i] = np.sqrt(vip_calc)
    return {feature:vip for vip,feature in zip(vips, X.columns.values)}

if __name__=='__main__':
    feature_sel = True
    n_components = 2
    n_perm = 10_000
    greenwitch_ref = True
    color_option = 'fa'

    climate_data = pd.read_csv('../source_data/wcr_extract_combined_chirps_agera_avg2015-2023_080825.csv').rename(columns={'id': 'site'})
    climate_data.columns = [col.strip().lower() for col in climate_data.columns]
    climate_data = climate_data.set_index('site')

    complementary_site_data = pd.read_csv('../source_data/site_info_shortestCoast_190925.csv').rename(columns={'id': 'site'}).set_index('site')
    complementary_site_data.columns = [col.strip().lower() for col in complementary_site_data.columns]
    complementary_site_data = complementary_site_data.rename(columns={'srtm': 'elevation'})[['elevation', 'distancecoast_km']]
    full_climate_data = climate_data.join(complementary_site_data, how='inner')

    feature_selections = {
        'rust': ["bioc_3", "vpd", "cdd", "bioc_12", "bioc_15", "bioc_17", "bioc_10", "bioc_1"],
        'yield': ["bioc_2", "vpd", "cdd", "bioc_6", "bioc_15", "bioc_12", "bioc_1", 'bioc_3'],
    }

    geo_features_to_add = ['northwardness', 'equatoriality']
    
    supplementary_variables = {
        'rust':  ['elevation', 'bioc_7', 't10', 'ndd', 'bioc_22', 'bioc_23'],
        'yield': ['elevation', 'bioc_4', 'bioc_7', 'ndd', 't10', 'bioc_22', 'bioc_25'],
    }
    
    xls_file_paths = {
        'rust': '../source_data/2025_WCR_master_results_Rust_Score_2025-08-27_.xlsx',
        'yield': '../source_data/2025_WCR_master_results_yield_2025-08-18_.xlsx',
    }
    drop_sites = {
        'rust': [],
        'yield': []
    }
    for path in ['./csv', './plot']:
        Path(path).mkdir(parents=True, exist_ok=True)

    result_dic = {}
    result_dic_perm = {}
    
    for outcome in [*feature_selections]:
        result_dic[outcome] = {}
        result_dic_perm[outcome] = {}
        xls_file_path = xls_file_paths[outcome]
        fa_data = pd.read_excel(xls_file_path, sheet_name='fa1_fa2_sites').set_index('site')
        
        joined_data = full_climate_data.join(fa_data, how='inner').reset_index()
        joined_data, geo_cols = add_geo_features(joined_data, 'latitude', 'longitude', greenwitch_ref=greenwitch_ref)
        if outcome in drop_sites and drop_sites[outcome]:
            drop_set = {str(s).lower() for s in drop_sites[outcome]}
            mask = ~joined_data['site'].astype(str).str.lower().isin(drop_set)
            joined_data = joined_data[mask].copy()

        joined_data["plot_site"] = joined_data["site"].astype(str).str.strip()

        feature_selection = feature_selections[outcome]
        supplementary_vars = supplementary_variables[outcome]
        supplementary_vars.extend(geo_features_to_add)
        climate_features = [f for f in feature_selection if f in joined_data.columns]
        
        pls = PLSRegression(n_components=n_components, scale=True)
        X = joined_data[climate_features]
        supplementary_data = joined_data[[col for col in supplementary_vars if col in joined_data.columns]]
        
        observed_variances_dicts = {}
        outcome_coeffs = {}
        vips = {}
        for fa in ['fa1', 'fa2']:
            y = joined_data[fa]
            pls.fit(X, y)
            coeffs = get_pls_reg_coeffs(pls, X=X, scaled=True)
            vip = get_vip(pls, X=X)
            vips[fa] = vip
            outcome_coeffs[fa] = coeffs
            observed_explained_variances = compute_explained_variance(pls, y)

            print(
                f"\nOutcome {outcome.upper()} - {fa}: "
                f"X variance explained: "
                f"t1={observed_explained_variances['X']['t1']:.3%}, "
                f"t2={observed_explained_variances['X']['t2']:.3%}, "
                f"X total={observed_explained_variances['X']['t1,t2']:.3%}; "
                f"y variance explained: "
                f"t1={observed_explained_variances['y']['t1']:.3%}, "
                f"t2={observed_explained_variances['y']['t2']:.3%}, "
                f"y total={observed_explained_variances['y']['t1,t2']:.3%}"
            )

            observed_variances_dicts[fa] = observed_explained_variances

            color_values = y if color_option == 'fa' else joined_data[color_option]
            color_label = fa if color_option == 'fa' else color_option

            fig, axs, supp_corrs = plot_correlation_circle(
                pls,
                feature_names=X.columns.values,
                target_name=fa,
                explained_var_dict=observed_explained_variances,
                rownames=joined_data["plot_site"],
                y_values=y,
                supplementary_vars=supplementary_vars,
                supplementary_data=supplementary_data,
                X_data=X,
                color_data=color_values,
                color_label=color_label
            )
            fig.savefig(f'./plot/pls_correlation_plot_{outcome}_{fa}.pdf', bbox_inches='tight')
            
            save_correlations_csv(
                pls_model=pls, 
                X_data=X, 
                y_values=y, 
                target_name=fa, 
                filename=f'./csv/{outcome}_{fa}_correlations.csv',
                supplementary_vars=supplementary_vars, 
                supplementary_data=supplementary_data
            )

            y_pred = pls.predict(X).flatten()
            spearman_rho = pd.Series(y_pred, index=y.index).corr(
                y, method="spearman"
            )
            print(
                f"Outcome {outcome.upper()} - {fa}: "
                f"Spearman rank corr. true vs predicted = {spearman_rho:.3f}"
            )
            fig, ax = plot_pred_vs_true(
                y, y_pred, fa, rownames=joined_data["plot_site"]
            )
            fig.savefig(f'./plot/pls_pred_vs_true_{outcome}_{fa}.pdf', bbox_inches='tight')
            
        coeff_filename =  f'./csv/{outcome}_pls_coefficients.csv'
        vip_filename =  f'./csv/{outcome}_pls_vips.csv'
        dict_to_csv(outcome_coeffs, coeff_filename)
        dict_to_csv(vips, vip_filename)
        perm_variances_dicts = {}
        pls_perm = PLSRegression(n_components=n_components, scale=True)
        for fa in ['fa1', 'fa2']:
            perm_variances_dict_list = []
            for _ in range(n_perm):
                y_perm = joined_data[fa].copy().sample(frac=1)
                pls_perm.fit(X, y_perm)
                perm_variances_dict = compute_explained_variance(pls_perm, y_perm)
                perm_variances_dict_list.append(perm_variances_dict)
            perm_variances_dict_combined = combine_dicts(perm_variances_dict_list)
            perm_variances_dicts[fa] = perm_variances_dict_combined

        p_values_dict = build_func_tree(observed_variances_dicts, perm_variances_dicts, get_p_value)
        observed_var_df = dict_to_metric_df(observed_variances_dicts, metric_name='explained_var')
        observed_var_df.to_csv(f'./csv/{outcome}_pls_explained_var.csv', index=False, float_format='%.4f')
        
        for fa in ['fa1', 'fa2']:
            print(f'\n Outcome {outcome.upper()} p_val {fa} y,(t1,t2): {p_values_dict[fa]["y"]["t1,t2"]}')
            
        pvalues_df = dict_to_metric_df(p_values_dict, metric_name='p_value')
        pvalues_df['explained_var'] = observed_var_df['explained_var']
        pvalues_df.to_csv(f'./csv/{outcome}_permutation_p_values.csv', index=False, float_format='%.5f')
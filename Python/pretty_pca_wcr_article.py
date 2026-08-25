import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CFG = {
    "fig_width_mm": 183,
    "fig_height_mm": 150,
    "dpi": 300,
    "use_varimax": True,
    "density_cmap": "Purples",
    "density_alpha": 0.55,
    "density_levels": 8,
    "density_thresh": 0.05,
    "wcr_color": "#1541A7",
    "wcr_marker": "o",
    "wcr_size": 28,
    "wcr_edgecolor": "white",
    "wcr_linewidth": 0.5,
    "arrow_color": "#FF0000",
    "arrow_alpha": 1.0,
    "arrow_scale": 3.2,
    "arrow_width": 0.055,
    "arrow_head_width": 0.18,
    "arrow_label_size": 10,
    "arrow_zorder": 10,
    "explicit_arrows": [
        "bioc_1",
        "bioc_2",
        "bioc_12",
        "bioc_7",
        "bioc_15",
        "bioc_3",
        "bioc_17",
    ],
    "n_arrows": 6,
    "use_adjust_text": True,
    "label_seed_offset": 0.12,
    "label_connector_color": "#666666",
    "label_connector_lw": 0.4,
    "axis_lim": 7,
    "grid_color": "#AAAAAA",
    "grid_lw": 0.5,
    "grid_ls": "--",
    "spine_lw": 0.8,
    "font_family": "sans-serif",
    "label_size": 9,
    "tick_size": 7,
    "annot_size": 8,
    "periods": ["Baseline", "Future"],
}

H2_SITES = {
    "Alstonville_AUS",
    "Buginyanya_UGA",
    "CCRI_IND",
    "CRI_ZWE",
    "Chanchamayo_PER",
    "Chicharras_MEX",
    "Cordoba_MEX",
    "FlorAmarilla_SLV",
    "Gahororo_RWA",
    "Gambung_IDN",
    "Kateshi_ZMB",
    "Koru_KEN",
    "LaCumplida_NIC",
    "LaFe_HND",
    "LaVirgen_NIC",
    "LasLagunas_HND",
    "Mulungu_COD",
    "Mwito_RWA",
    "Mzuzu_MWI",
    "Paksong_LAO",
    "Rubona_RWA",
    "Ruiru_KEN",
    "SanAntonio_GTM",
    "SanIgnacio_PER",
    "Sumatra_IDN",
    "Teocelo_MEX",
    "Toraja_IDN",
}

MM_TO_IN = 1 / 25.4

path = "../source_data/bunn-trials_aclimatar-datasets_bsl-ftr_april1426.csv"
df = pd.read_csv(path)
df["id"] = df["id"].astype(str).str.strip()
df["source"] = df["source"].astype(str).str.strip()
df["period"] = df["period"].astype(str).str.strip()

bioc_cols = [c for c in df.columns if c.startswith("bioc_")] + ["t10", "cdd"]


def get_subset(data, source, period):
    sub = data[(data["source"] == source) & (data["period"] == period)].copy()
    sub[bioc_cols] = sub[bioc_cols].apply(pd.to_numeric, errors="coerce")
    return sub.dropna(subset=bioc_cols)


def varimax(loadings, max_iter=1000, tol=1e-6):
    p, k = loadings.shape
    rotation = np.eye(k)
    var_old = 0.0
    for _ in range(max_iter):
        lam = loadings @ rotation
        u, _, vt = np.linalg.svd(
            loadings.T
            @ (
                lam**3
                - (1.0 / p) * lam @ np.diag(np.sum(lam**2, axis=0))
            )
        )
        rotation = u @ vt
        var_new = np.sum(np.var(lam**2, axis=0))
        if abs(var_new - var_old) < tol:
            break
        var_old = var_new
    return loadings @ rotation, rotation


base_df = get_subset(df, "Bunn", "Baseline")
if base_df.empty:
    raise ValueError("No usable rows for source='Bunn', period='Baseline'.")

scaler = StandardScaler()
x_base = scaler.fit_transform(base_df[bioc_cols])

pca = PCA(n_components=2)
pca.fit(x_base)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
total_variance = x_base.var(axis=0, ddof=1).sum()

if CFG["use_varimax"]:
    rotated_loadings, r_varimax = varimax(loadings)
    rot_variances = np.sum(rotated_loadings**2, axis=0)
    rotation_note = " (varimax)"
    pc_prefix = "RC"
else:
    rotated_loadings, r_varimax = loadings, np.eye(2)
    rot_variances = pca.explained_variance_
    rotation_note = ""
    pc_prefix = "PC"

rot_exp_var = (rot_variances / total_variance) * 100

if CFG["explicit_arrows"]:
    top_idx = [
        i for i, name in enumerate(bioc_cols) if name in CFG["explicit_arrows"]
    ]
else:
    top_idx = np.argsort(np.linalg.norm(loadings, axis=1))[-CFG["n_arrows"]:]

wcr_base = get_subset(df, "WCR", "Baseline")
wcr_base = wcr_base[wcr_base["id"].isin(H2_SITES)].copy()
wcr_base["plot_id"] = wcr_base["id"]
if wcr_base.empty:
    raise ValueError("No usable WCR baseline rows for the selected H2 sites.")
wcr_pca_rot = pca.transform(scaler.transform(wcr_base[bioc_cols])) @ r_varimax

plt.rcParams.update(
    {
        "font.family": CFG["font_family"],
        "axes.linewidth": CFG["spine_lw"],
        "xtick.labelsize": CFG["tick_size"],
        "ytick.labelsize": CFG["tick_size"],
        "xtick.major.width": CFG["spine_lw"],
        "ytick.major.width": CFG["spine_lw"],
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig_w = CFG["fig_width_mm"] * MM_TO_IN
fig_h = CFG["fig_height_mm"] * MM_TO_IN
fig_cc, ax_cc = plt.subplots(figsize=(fig_h, fig_h))
circle = plt.Circle(
    (0, 0),
    1,
    color=CFG["grid_color"],
    fill=False,
    ls=CFG["grid_ls"],
    lw=CFG["grid_lw"],
)
ax_cc.add_patch(circle)

for i, col in enumerate(bioc_cols):
    ax_cc.annotate(
        "",
        xy=(rotated_loadings[i, 0], rotated_loadings[i, 1]),
        xytext=(0, 0),
        arrowprops={
            "arrowstyle": (
                f"->,head_width={CFG['arrow_head_width'] * 0.5},"
                "head_length=0.12"
            ),
            "color": CFG["arrow_color"],
            "lw": CFG["arrow_width"] * 15,
            "alpha": CFG["arrow_alpha"],
        },
        zorder=CFG["arrow_zorder"],
    )
    ax_cc.text(
        rotated_loadings[i, 0] * 1.15,
        rotated_loadings[i, 1] * 1.15,
        col,
        color="#000000",
        ha="center",
        va="center",
        fontsize=CFG["arrow_label_size"],
        fontweight="bold",
        zorder=999,
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
    )

for fn in (ax_cc.axhline, ax_cc.axvline):
    fn(0, color=CFG["grid_color"], ls=CFG["grid_ls"], lw=CFG["grid_lw"])

ax_cc.set_xlim(-1.25, 1.25)
ax_cc.set_ylim(-1.25, 1.25)
ax_cc.set_aspect("equal")
ax_cc.set_xlabel(
    f"{pc_prefix}1 ({rot_exp_var[0]:.1f}%){rotation_note}",
    fontsize=CFG["label_size"],
)
ax_cc.set_ylabel(
    f"{pc_prefix}2 ({rot_exp_var[1]:.1f}%){rotation_note}",
    fontsize=CFG["label_size"],
)
ax_cc.tick_params(direction="out")
sns.despine(ax=ax_cc)

fig_cc.tight_layout()
fig_cc.savefig(
    "./plots/correlation_circle.pdf",
    dpi=CFG["dpi"],
    bbox_inches="tight",
)
plt.close(fig_cc)


def period_label(period):
    if period == "Baseline":
        return "Current"
    if period == "Future":
        return "Projected (averaged over 2025-2055)"
    return period


def draw_biplot(ax, period, show_ylabel=True):
    bunn = get_subset(df, "Bunn", period)
    if len(bunn) > 1:
        bpca_rot = pca.transform(scaler.transform(bunn[bioc_cols])) @ r_varimax
        sns.kdeplot(
            x=bpca_rot[:, 0],
            y=bpca_rot[:, 1],
            ax=ax,
            cmap=CFG["density_cmap"],
            fill=True,
            alpha=CFG["density_alpha"],
            thresh=CFG["density_thresh"],
            levels=CFG["density_levels"],
        )

    ax.scatter(
        wcr_pca_rot[:, 0],
        wcr_pca_rot[:, 1],
        c=CFG["wcr_color"],
        marker=CFG["wcr_marker"],
        s=CFG["wcr_size"],
        edgecolors=CFG["wcr_edgecolor"],
        linewidths=CFG["wcr_linewidth"],
        zorder=5,
    )

    norms = np.linalg.norm(wcr_pca_rot, axis=1, keepdims=True).clip(min=1e-6)
    site_texts = [
        ax.text(
            wcr_pca_rot[i, 0]
            + CFG["label_seed_offset"] * wcr_pca_rot[i, 0] / norms[i, 0],
            wcr_pca_rot[i, 1]
            + CFG["label_seed_offset"] * wcr_pca_rot[i, 1] / norms[i, 0],
            label,
            fontsize=CFG["annot_size"],
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        for i, label in enumerate(wcr_base["id"])
    ]

    scale = CFG["arrow_scale"]
    bioc_texts = []
    for i in top_idx:
        ax.annotate(
            "",
            xy=(
                rotated_loadings[i, 0] * scale,
                rotated_loadings[i, 1] * scale,
            ),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": (
                    f"->,head_width={CFG['arrow_head_width']},"
                    "head_length=0.12"
                ),
                "color": CFG["arrow_color"],
                "lw": CFG["arrow_width"] * 20,
                "alpha": CFG["arrow_alpha"],
            },
            zorder=CFG["arrow_zorder"],
        )
        bioc_texts.append(
            ax.text(
                rotated_loadings[i, 0] * scale * 1.15,
                rotated_loadings[i, 1] * scale * 1.15,
                bioc_cols[i],
                color="#000000",
                ha="center",
                va="center",
                fontsize=CFG["arrow_label_size"],
                fontweight="bold",
                zorder=999,
                path_effects=[
                    pe.withStroke(linewidth=1.5, foreground="white")
                ],
            )
        )

    if CFG["use_adjust_text"]:
        adjust_text(
            site_texts,
            add_objects=bioc_texts,
            x=wcr_pca_rot[:, 0],
            y=wcr_pca_rot[:, 1],
            ax=ax,
            expand_text=(1.0, 1.0),
            expand_points=(1.0, 1.0),
            force_text=0.005,
            force_points=0.2,
            min_arrow_len=0,
            arrowprops={
                "arrowstyle": "-",
                "color": CFG["label_connector_color"],
                "lw": CFG["label_connector_lw"],
            },
        )

    for fn in (ax.axhline, ax.axvline):
        fn(
            0,
            color=CFG["grid_color"],
            ls=CFG["grid_ls"],
            lw=CFG["grid_lw"],
            zorder=1,
        )

    ax.set_xlim(-CFG["axis_lim"], CFG["axis_lim"])
    ax.set_ylim(-CFG["axis_lim"], CFG["axis_lim"])
    ax.set_xlabel(
        f"{pc_prefix}1 ({rot_exp_var[0]:.1f}%){rotation_note}",
        fontsize=CFG["label_size"],
    )
    if show_ylabel:
        ax.set_ylabel(
            f"{pc_prefix}2 ({rot_exp_var[1]:.1f}%){rotation_note}",
            fontsize=CFG["label_size"],
        )
    ax.tick_params(direction="out")
    sns.despine(ax=ax, trim=True)

    handles = [
        mpatches.Patch(
            color=plt.get_cmap(CFG["density_cmap"])(0.65),
            alpha=CFG["density_alpha"],
            label=f"C. Arabica presence points - {period_label(period)}",
        ),
        mpatches.Patch(
            color=CFG["wcr_color"],
            label="IMLVT trial sites (Current)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=CFG["tick_size"],
    )


for period in CFG["periods"]:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_biplot(ax, period)
    fig.tight_layout()
    fig.savefig(
        f"./plots/pca_{period.lower()}.pdf",
        dpi=CFG["dpi"],
        bbox_inches="tight",
    )
    plt.close(fig)

fig_comb, axes = plt.subplots(
    1,
    2,
    figsize=(fig_w * 1.8, fig_h),
    sharex=True,
    sharey=True,
)
for ax, period in zip(axes, CFG["periods"]):
    draw_biplot(ax, period, show_ylabel=(ax == axes[0]))
    ax.set_title(
        period_label(period),
        fontsize=CFG["label_size"] + 2,
        fontweight="bold",
    )
fig_comb.tight_layout()
fig_comb.savefig(
    "./plots/trial_sites_climate_representativeness.pdf",
    dpi=CFG["dpi"],
    bbox_inches="tight",
)
plt.close(fig_comb)

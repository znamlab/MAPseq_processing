from scipy.stats import pearsonr
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from final_processing.MAPseq_data_processing import FitResult
import final_processing.figure_formatting as ff
import final_processing.helper_functions as hf
import final_processing.MAPseq_data_processing as mdp
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list


# def bubble_plot(dataframe, pval_df, subplot_size, font_size, size_scale=30):
#     """
#     Create a bubble plot to visualize log-ratios with p-values.
#     Args:
#     - dataframe: pd.DataFrame, shape (n_rows, n_cols)
#         DataFrame of log-ratios (log₁₀(observed / expected)).
#     - pval_df: pd.DataFrame, shape (n_rows, n_cols)
#         DataFrame of p-values for each log-ratio.
#     - size_scale: int, default 100
#         Scaling factor for bubble sizes.
#     """
#     plt.rcParams["font.family"] = "Arial"

#     if dataframe.shape != pval_df.shape:
#         raise ValueError("dataframe and pval_df must have the same shape.")

#     pval_df = pval_df.reindex_like(dataframe)
#     df_plot = dataframe.stack(dropna=False).reset_index()
#     df_plot.columns = ["y_label", "x_label", "conditional_prob"]
#     df_plot["p_value"] = pval_df.stack(dropna=False).reset_index(drop=True)
#     x_categories = dataframe.columns
#     y_categories = dataframe.index

#     df_plot["x"] = pd.Categorical(
#         df_plot["x_label"], categories=x_categories, ordered=True
#     ).codes
#     df_plot["y"] = pd.Categorical(
#         df_plot["y_label"], categories=y_categories, ordered=True
#     ).codes
#     df_plot["bubble_size"] = df_plot["conditional_prob"].abs() * size_scale
#     df_plot["color_value"] = -np.log10(df_plot["p_value"].clip(lower=1e-50))
#     purples_cmap = plt.cm.Purples
#     new_colors = purples_cmap(np.linspace(0.15, 1, 256))
#     custom_purples = mcolors.LinearSegmentedColormap.from_list(
#         "custom_purples", new_colors
#     )

#     fig, ax = plt.subplots(figsize=(subplot_size[0], subplot_size[1]))
#     sc = ax.scatter(
#         x=df_plot["x"],
#         y=df_plot["y"],
#         s=df_plot["bubble_size"],
#         c=df_plot["color_value"],
#         cmap=custom_purples,
#         vmin=df_plot["color_value"].min() * 0.8,
#         vmax=df_plot["color_value"].max(),
#         edgecolors="none",
#     )
#     df_plot["significant"] = df_plot["p_value"] < 0.05
#     df_signif = df_plot[df_plot["significant"]]

#     ax.scatter(
#         x=df_signif["x"],
#         y=df_signif["y"],
#         s=df_signif["bubble_size"],
#         facecolors="none",
#         edgecolors="black",
#         linewidths=0.5,
#     )

#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("-log10(p-value)", fontsize=5)
#     cbar.ax.tick_params(labelsize=5)
#     legend_values = [0.1, 0.4, 0.8]
#     legend_handles = [
#         ax.scatter(
#             [],
#             [],
#             s=val * size_scale,
#             c="gray",
#             alpha=0.5,
#             label=rf"$\mathit{{P(target \mid VC\ area)}}$ = {val}",
#         )
#         for val in legend_values
#     ]

#     cbar_ax = cbar.ax
#     box = cbar_ax.get_position()
#     legend_x = box.x1 + 0.45

#     legend = ax.legend(
#         handles=legend_handles,
#         title="Conditional Probability",
#         loc="upper left",
#         bbox_to_anchor=(legend_x, 1),
#         borderaxespad=0.0,
#         frameon=True,
#         handleheight=2.0,
#         fontsize=font_size,
#         title_fontsize=font_size,
#     )
#     ax.set_xticks(range(len(x_categories)))
#     ax.set_yticks(range(len(y_categories)))
#     ax.set_xticklabels(x_categories, rotation=90, fontsize=font_size)
#     ax.set_yticklabels(y_categories, fontsize=font_size)
#     ax.invert_yaxis()

#     plt.xlabel("Co-Projection Target", fontsize=font_size, fontweight="bold")
#     plt.ylabel("VC area", fontsize=font_size, fontweight="bold")
#     plt.tight_layout()
#     plt.show()
#     return fig


def plot_exponential_fit(ax, fit: FitResult):
    ax.plot(
        fit.fitted_x,
        fit.fitted_y,
        color="black",
        lw=0.5,
        ls="dotted",
        label="Exponential fit\nℓ = " f"{1/fit.params[1]:.0f} µm",
    )


def plot_errorbars(ax, distances, means, errors, color_map):
    for area in means.index:
        ax.errorbar(
            distances[area],
            means[area],
            yerr=errors[area],
            fmt="o",
            color="black",
            mfc=color_map.get(area, "lightgray"),
            mec=color_map.get(area, "lightgray"),
            ms=2,
            elinewidth=0.5,
            capsize=0.5,
        )


def plot_ap_vs_visap(ax, data, gen_parameters):
    """
    function to draw AP-Vis-to-soma scatter plot and regression on supplied ax and return rho and p-value.
    """
    plt.rcParams["text.usetex"] = False
    convert_dict = hf.get_convert_dict()
    data = data.copy()
    data["converted"] = data["VC_majority"].map(convert_dict)
    palette = hf.get_colour_dict(allen_nomenclature=False)
    rho, pval = pearsonr(data["AP_Vis"], data["mean_AP_soma"])
    sb.regplot(
        x="AP_Vis",
        y="mean_AP_soma",
        data=data,
        color="black",
        scatter=False,
        ci=95,
        scatter_kws={"s": 5},
        line_kws={"linewidth": 1},
        ax=ax,
    )
    scatter = sb.scatterplot(
        x="AP_Vis",
        y="mean_AP_soma",
        data=data,
        hue="converted",
        legend=False,
        palette=palette,
        s=5,
        ax=ax,
    )
    symbol_add, exp_add = ff.convert_to_exp(num=pval)

    ax.text(
        0.75,
        0.25,
        f"r={rho:.3f}\np{symbol_add}10$^{{{exp_add}}}$",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=gen_parameters["font_size"],
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
    )
    xlabel = "VC cubelet A-P\nposition (µm)"
    ylabel = "Mean soma A-P\nposition (µm)"
    ax.text(
        1.45,
        -0.3,
        "Anterior",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=gen_parameters["font_size"] * 0.9,
    )
    ax.text(
        -0.4,
        1.1,
        "Anterior",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=gen_parameters["font_size"] * 0.9,
    )
    ax.text(
        -0.4,
        -0.3,
        "Posterior",
        transform=ax.transAxes,
        fontsize=gen_parameters["font_size"] * 0.9,
        va="bottom",
        ha="center",
    )
    ff.myPlotSettings_splitAxis(
        fig=ax.figure,
        ax=ax,
        ytitle=ylabel,
        xtitle=xlabel,
        title="",
        mySize=gen_parameters["font_size"],
    )
    ax.set_xticks([0, 3000])


def plot_indiv_distance_decay(
    ax,
    projections_df,
    distances,
    font_size,
):
    """
    function to plot distance-decay plot with exponential fit.
    """
    means, errors = mdp.get_means_errors(projections_df)
    fit_obj = None
    if (
        len(distances) > 1
        and (means > 0).any()
        and not distances.isna().any()
        and not means.isna().any()
    ):
        fit_obj = mdp.fit_exponential(distances, means)
        plot_exponential_fit(ax, fit_obj)
        print(
            f"KS = {fit_obj.ks_stat:.2f} (p={fit_obj.ks_p:.3g}), "
            f"r  = {fit_obj.cv_corr:.2f} (p={fit_obj.cv_p:.3g})"
        )
    ff.plot_errorbars(
        ax,
        distances,
        means,
        errors,
        color_map=hf.get_colour_dict(allen_nomenclature=True),
    )
    ff.myPlotSettings_splitAxis(
        fig=ax.figure,
        ax=ax,
        ytitle="",
        xtitle="Distance (µm)",
        title="",
        mySize=font_size,
    )
    return fit_obj


def plot_all_distance_decay_from_A1(
    fig,
    axes,
    *,
    combined_dict,
    all_mice_combined,
    mice,
    gen_parameters,
):
    """
    generate the two distance-decay plots in fig. 2
    """
    plt.subplots_adjust(wspace=1)

    freq_df, freq_df_strength, distances = mdp.get_distances_from_A1(
        combined_dict=combined_dict,
        area_cols=all_mice_combined.columns,
        mice=mice,
    )

    for ax, df in zip(axes, [freq_df, freq_df_strength]):
        plot_indiv_distance_decay(
            ax,
            df,
            distances,
            font_size=gen_parameters["font_size"],
        )

    ff.style_figure_exp_fits(
        fig,
        axes,
        color_map=hf.get_colour_dict(allen_nomenclature=True),
        convert_dict=hf.get_convert_dict(),
        font_size=gen_parameters["font_size"],
    )


def plot_area_AP_positions(
    ax,
    fig,
    area_AP_dict,
    where_AP_vis,
    font_size,
):
    """
    plots A-P soma positions by area with mice as points plus the means
    """
    convert_dict = hf.get_convert_dict()
    which_colour = hf.get_colour_dict(allen_nomenclature=True)
    keys_sorted = sorted(area_AP_dict.keys(), key=lambda k: where_AP_vis[k])
    for i, key in enumerate(keys_sorted):
        positions = area_AP_dict[key]
        values = [val * 25 for val in positions]
        color = which_colour[key]

        xvals = np.full(len(values), i)
        ax.scatter(
            xvals,
            values,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidth=0.5,
            zorder=1,
            alpha=0.3,
            s=10,
            label=key,
        )
        mean_val = np.mean(values)
        # ax.hlines(
        #     y=mean_val,
        #     xmin=i - 0.4,
        #     xmax=i + 0.4,
        #     colors=color,
        #     linewidth=2,
        # )
        ax.plot(
            [i - 0.25, i + 0.25],
            [mean_val, mean_val],
            color=color,
            linewidth=2,
            zorder=2,
        )
    converted_labels = [convert_dict.get(k, k) for k in keys_sorted]
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle="",
        xtitle="",
        title="",
        mySize=font_size,
    )
    ax.set_xticks(range(len(keys_sorted)))
    ax.set_xticklabels(converted_labels, rotation=90, size=font_size)
    ax.set_ylabel("Mean soma A-P\nposition (µm)", size=font_size)
    ax.set_ylim(bottom=2000, top=2600)
    plt.tight_layout()


def plot_projection_probability_by_area(
    ax,
    area,
    proj_by_ap_pos_df,
    model,
    font_size,
    pval,
    idx,
    total_axes,
    num_bins=8,
    dist_only=False,
    include_pval=False,
):
    """
    plot projection probability vs A-P position for each area in fig 2.
    """
    if dist_only:
        variable_to_plot = "distance"
    else:
        variable_to_plot = "AP_position"
    convert_dict = hf.get_convert_dict()
    color_dict = hf.get_colour_dict(allen_nomenclature=True)
    df_area = proj_by_ap_pos_df[proj_by_ap_pos_df["Area"] == area].copy()
    ap_vals = df_area[f"{variable_to_plot}"].values
    proj_vals = df_area["Projection"].values

    bin_edges = np.linspace(ap_vals.min(), ap_vals.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_proportions = []

    for i in range(num_bins):
        in_bin = (ap_vals >= bin_edges[i]) & (ap_vals < bin_edges[i + 1])
        if np.sum(in_bin) > 0:
            bin_proportions.append(proj_vals[in_bin].mean())
        else:
            bin_proportions.append(np.nan)

    valid = ~np.isnan(bin_proportions)
    main_color = color_dict.get(area, "black")
    fit_color = ff.adjust_color(main_color, 0.7)
    ci_color = ff.adjust_color(main_color, 1.3)

    ax.plot(
        bin_centers[valid],
        np.array(bin_proportions)[valid],
        "o",
        label="Binned data",
        color=main_color,
        markersize=2,
        alpha=0.7,
    )

    ap_grid = np.linspace(ap_vals.min(), ap_vals.max(), 100)
    new_data = pd.DataFrame({f"{variable_to_plot}": ap_grid})
    pred = model.get_prediction(new_data)
    pred_df = pred.summary_frame(alpha=0.05)

    ax.plot(ap_grid, pred_df["predicted"], "-", color=fit_color, linewidth=1)
    ax.fill_between(
        ap_grid, pred_df["ci_lower"], pred_df["ci_upper"], color=ci_color, alpha=0.3
    )
    if dist_only:
        xtitle = "Euclidean distance"
    else:
        xtitle = "Soma A-P position"

    if include_pval:
        ff.myPlotSettings_splitAxis(
            fig=ax.figure,
            ax=ax,
            ytitle="Projection probability",
            xtitle=xtitle,
            title="",
            axisColor="k",
            mySize=font_size,
        )

        print(f"p value for {area} is {pval:.2g}")
        text_anchor = ("left", 0.05) if idx < 2 else ("right", 0.95)
        ax.text(
            text_anchor[1],
            1,
            f"{convert_dict.get(area, area)}\np = {pval:.2g}",
            transform=ax.transAxes,
            ha=text_anchor[0],
            va="top",
            fontsize=font_size,
            color="black",
        )
    else:
        ff.myPlotSettings_splitAxis(
            fig=ax.figure,
            ax=ax,
            ytitle="",
            xtitle=xtitle if idx == total_axes - 1 else "",
            title="",
            axisColor="k",
            mySize=font_size,
        )

        print(f"p value for {area} is {pval:.2g}")
        text_anchor = ("left", 0.05) if idx < 2 else ("right", 0.95)
        ax.text(
            text_anchor[1],
            1,
            convert_dict.get(area, area),
            transform=ax.transAxes,
            ha=text_anchor[0],
            va="top",
            fontsize=font_size,
            color="black",
        )
    _, ymax = ax.get_ylim()
    rounded_ymax = round(ymax, 1)
    ax.set_ylim(0, rounded_ymax)
    ax.set_yticks([0, rounded_ymax])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if not dist_only:
        ax.set_xticks([2000, 2500, 3000])


def plot_area_projection_probs(
    areas_to_plot,
    gen_parameters,
    combined_dict,
    AP_position_dict_list_combined,
    fig,
    axes,
    include_dist_as_covariate=False,
    only_plot_distance=False,
    include_pval=False,
):
    """
    plot projection probability vs A-P position for all areas specified in list in fig 2.
    """
    (
        pval_df,
        proj_by_ap_pos_df,
        results_population_dict,
    ) = mdp.individual_area_probabilities(
        gen_parameters=gen_parameters,
        combined_dict=combined_dict,
        AP_position_dict_list_combined=AP_position_dict_list_combined,
        include_distance=include_dist_as_covariate,
        distance_only=only_plot_distance,
    )
    axes = axes.flatten()
    if include_pval:
        for idx, (ax, area) in enumerate(zip(axes, areas_to_plot)):
            model = results_population_dict[area]
            pval = pval_df.loc[area, "p_value_corrected"]
            plot_projection_probability_by_area(
                ax=ax,
                area=area,
                proj_by_ap_pos_df=proj_by_ap_pos_df,
                model=model,
                font_size=gen_parameters["font_size"],
                pval=pval,
                idx=idx,
                total_axes=len(axes),
                dist_only=only_plot_distance,
                include_pval=include_pval,
            )
    else:
        for idx, (ax, area) in enumerate(zip(axes, areas_to_plot)):
            model = results_population_dict[area]
            pval = pval_df.loc[area, "p_value_corrected"]
            plot_projection_probability_by_area(
                ax=ax,
                area=area,
                proj_by_ap_pos_df=proj_by_ap_pos_df,
                model=model,
                font_size=gen_parameters["font_size"],
                pval=pval,
                idx=idx,
                total_axes=len(axes),
                dist_only=only_plot_distance,
            )

        fig.supylabel(
            "Projection probability",
            fontsize=gen_parameters["font_size"],
            x=0.1,
        )
    plt.tight_layout()
    plt.show()


def plot_ML_positioning_soma_vs_VC_cubelet(gen_parameters, fig, ax):
    """Function to plot fig. S4a ML positioning correlation between mean ML soma position of VC projecting neurons and ML VC cubelet position"""
    rho, pval, ML_soma_VC_sample = mdp.analyse_AC_VC_ML_correlation(gen_parameters)
    font_size = gen_parameters["font_size"]
    sb.regplot(
        x="ML_Vis",
        y="mean_ML_soma",
        data=ML_soma_VC_sample,
        color="black",
        scatter=False,
        ci=95,
        scatter_kws={"s": 5},
        line_kws={"linewidth": 1},
    )
    scatter = sb.scatterplot(
        x="ML_Vis",
        y="mean_ML_soma",
        data=ML_soma_VC_sample,
        hue="converted",
        palette=hf.get_colour_dict(allen_nomenclature=False),
        s=5,
        legend=True,
    )

    handles, labels = scatter.get_legend_handles_labels()

    ax.legend(
        handles=handles,
        frameon=False,
        labels=labels,
        title="Main VC area",
        bbox_to_anchor=(1.3, -0.2),
        loc="lower left",
        borderaxespad=0,
        fontsize=font_size,
        handlelength=1,
        handletextpad=0.4,
        markerscale=0.2,
    )
    leg = ax.get_legend()  # the auto legend
    leg.set_title("Main VC area", prop={"size": font_size})
    for txt in leg.get_texts():
        txt.set_fontsize(font_size)
    plt.text(
        0.8,
        0.2,
        f"r = {rho:.2f}\np = {pval:.2e}",
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=font_size,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
    )
    xlabel = "VC cubelet M-L position (µm)"
    ylabel = "Mean soma A-P\nposition (µm)"
    plt.text(
        1.25,
        -0.2,
        "Lateral",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=font_size,
    )
    plt.text(
        -0.1,
        1.15,
        "Lateral",
        ha="center",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=font_size,
    )
    plt.text(
        -0.1,
        -0.2,
        "Medial",
        transform=plt.gca().transAxes,
        fontsize=font_size,
        va="bottom",
        ha="center",
    )
    ff.myPlotSettings_splitAxis(
        fig=fig, ax=ax, ytitle=ylabel, xtitle=xlabel, title="", mySize=font_size
    )
    plt.tight_layout()
    plt.show()


def plot_vc_projecting_cells(visual_areas, all_mice, fig, ax, gen_parameters):
    """
    Fig. 2h. Plots the proportion of VC-projecting cells targeting different numbers of visual areas.
    """
    vis_areas_per_neuron = mdp.get_VIS_co_proj_distrib(
        visual_areas=visual_areas, all_mice=all_mice
    )
    counts = np.bincount(vis_areas_per_neuron, minlength=11)
    ax.bar(
        np.arange(11)[1:], counts[1:] / np.sum(counts), color="black", edgecolor="black"
    )
    ax.set_xticks([1, 10])
    ax.set_yticks([0.0, 0.2, 0.4])

    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle="Proportion of\nVC-projecting cells",
        xtitle="# Visual areas targeted",
        title="",
        mySize=gen_parameters["font_size"],
    )


def plot_max_counts_vs_areas_targeted(
    fig, ax, gen_parameters, combined_dict, all_mice_combined
):
    """
    supp figure showing how max cubelet barcode count per barcode changes with number of visual areas targeted
    """
    vis_areas_per_neuron = mdp.get_VIS_co_proj_distrib(
        visual_areas=gen_parameters["HVA_cols"], all_mice=all_mice_combined
    )
    max_counts = mdp.get_max_counts(
        combined_dict=combined_dict,
        gen_parameters=gen_parameters,
        all_mice_combined=all_mice_combined,
    )
    log_max = np.log2(max_counts)
    bin_width = 2
    bins = np.arange(
        np.floor(log_max.min()), np.ceil(log_max.max()) + bin_width, bin_width
    )
    bin_centers = bins[:-1] + bin_width / 2
    df = pd.DataFrame({"log_max": log_max, "n_areas": vis_areas_per_neuron})
    df["x_bin"] = pd.cut(df["log_max"], bins=bins, labels=bin_centers)

    bar_height = df.groupby("x_bin")["n_areas"].mean().reindex(bin_centers)
    ax.bar(
        bin_centers,
        bar_height,
        width=bin_width * 0.8,
        color="purple",
        alpha=0.6,
        edgecolor="black",
    )
    ax.scatter(
        log_max, vis_areas_per_neuron, s=10, color="black", alpha=0.15, linewidth=0
    )
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle="Visual areas targeted",
        xtitle="Log$_{2}$(max count)",
        title="",
        mySize=gen_parameters["font_size"],
    )
    ax.set_xlim(bins.min() - bin_width * 0.5, bins.max() + bin_width * 0.5)
    # bin_labels = [f"{int(b)}-{int(b+bin_width)}" for b in bins[:-1]]
    # x_binned = pd.cut(
    #     log_max, bins=bins, labels=bin_labels, include_lowest=True, right=False
    # )
    # # df_to_plot = pd.DataFrame({"x": vis_areas_per_neuron, "y": np.log2(max_counts)})
    # df_to_plot = pd.DataFrame({"x": x_binned, "y": vis_areas_per_neuron})
    # sb.barplot(x="x", y="y", data=df_to_plot, alpha=0.6, color="purple", ci=None)
    # sb.stripplot(x="x", y="y", data=df_to_plot, color="black", alpha=0.1, jitter=True)
    # ff.myPlotSettings_splitAxis(
    #     fig=fig,
    #     ax=ax,
    #     ytitle="Visual areas targeted",
    #     xtitle="# Log$_{2}$(max count)",
    #     title="",
    #     mySize=gen_parameters["font_size"],
    # )


def plot_supp_cond_prob_heatmaps(gen_parameters, axs, all_mice_combined):
    proj_path = gen_parameters["proj_path"]
    p_val_adj_matrix, conditional_probability_dict = mdp.get_p_val_comp_to_shuffled(
        proj_path=proj_path, all_mice_combined=all_mice_combined, comp_VIS_only=False
    )
    font_size = gen_parameters["font_size"]
    heatmap_titles = ["Observed", "Mean shuffled"]
    cbar_label = "Conditional probability\nP(target|cortical area"
    col_order_used = conditional_probability_dict["observed"].columns
    combined_dif = np.log2(
        conditional_probability_dict["observed"].loc[col_order_used]
        / conditional_probability_dict["shuffled"].loc[col_order_used]
    )
    combined_dif_converted = hf.convert_matrix_names(combined_dif)
    dfs = [
        conditional_probability_dict["observed"].loc[col_order_used],
        conditional_probability_dict["shuffled"].loc[col_order_used],
    ]
    for number, title in enumerate(heatmap_titles):
        data_to_use = dfs[number].copy(deep=True)
        np.fill_diagonal(data_to_use.values, 0)
        sb.heatmap(
            ax=axs[number],
            data=hf.convert_matrix_names(data_to_use),
            cmap="Purples",
            xticklabels=True,
            yticklabels=True,
            cbar_kws=dict(
                location="right",
                pad=-0.35,
                fraction=0.045,
                shrink=1.0,
                use_gridspec=True,
            ),
        )
        axs[number].set_title(f"{heatmap_titles[number]}", size=font_size)
        axs[number].tick_params(
            axis="y", which="major", labelsize=font_size, rotation=0
        )

        for _, spine in axs[number].spines.items():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        axs[number].set_ylabel("Cortical area", size=font_size)
        axs[number].set_xlabel("Co-projection target", size=font_size)

        cbar = axs[number].collections[0].colorbar

        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)
        cbar.set_label(cbar_label, fontsize=font_size)

    pval_df = hf.convert_matrix_names(p_val_adj_matrix.loc[col_order_used])
    size_scale = 15
    combined_dif_converted_df_plot = combined_dif_converted.stack(
        dropna=False
    ).reset_index()
    combined_dif_converted_df_plot.columns = [
        "y_label",
        "x_label",
        "subtracted_conditional_prob",
    ]
    combined_dif_converted_df_plot["p_value"] = pval_df.stack(dropna=False).reset_index(
        drop=True
    )
    x_categories = combined_dif_converted.columns
    y_categories = combined_dif_converted.index
    combined_dif_converted_df_plot["x"] = pd.Categorical(
        combined_dif_converted_df_plot["x_label"], categories=x_categories, ordered=True
    ).codes
    combined_dif_converted_df_plot["y"] = pd.Categorical(
        combined_dif_converted_df_plot["y_label"], categories=y_categories, ordered=True
    ).codes
    combined_dif_converted_df_plot["bubble_size"] = (
        combined_dif_converted_df_plot["subtracted_conditional_prob"].abs() * size_scale
    )
    combined_dif_converted_df_plot["color_value"] = np.sign(
        combined_dif_converted_df_plot["subtracted_conditional_prob"]
    ) * -np.log10(combined_dif_converted_df_plot["p_value"].clip(lower=1e-5))
    norm = TwoSlopeNorm(
        vmin=combined_dif_converted_df_plot["color_value"].min(),
        vcenter=0,
        vmax=combined_dif_converted_df_plot["color_value"].max(),
    )

    sc = axs[2].scatter(
        x=combined_dif_converted_df_plot["x"],
        y=combined_dif_converted_df_plot["y"],
        s=combined_dif_converted_df_plot["bubble_size"],
        c=combined_dif_converted_df_plot["color_value"],
        cmap="coolwarm",
        norm=norm,
        edgecolors="none",
    )
    significant_mask = combined_dif_converted_df_plot["p_value"] < 0.05
    axs[2].scatter(
        x=combined_dif_converted_df_plot.loc[significant_mask, "x"],
        y=combined_dif_converted_df_plot.loc[significant_mask, "y"],
        s=combined_dif_converted_df_plot.loc[significant_mask, "bubble_size"],
        facecolors="none",
        edgecolors="black",
        linewidths=1,
    )
    axs[2].set_xticks(range(len(x_categories)))
    axs[2].set_yticks(range(len(y_categories)))
    axs[2].set_xticklabels(x_categories, rotation=90, fontsize=font_size * 0.9)
    axs[2].set_yticklabels(y_categories, fontsize=font_size * 0.9)

    axs[2].invert_yaxis()
    axs[2].set_title("Effect size (observed/shuffled)", size=font_size)
    axs[2].set_xlabel("Co-projection target", fontsize=font_size * 0.9)
    axs[2].set_ylabel("Cortical area", fontsize=font_size * 0.9)
    legend_values = [0.1, 0.5, 1.5]
    legend_handles = [
        axs[2].scatter(
            [], [], s=val * size_scale, c="gray", alpha=0.5, label=f"{val:.1f}"
        )
        for val in legend_values
    ]

    axs[2].legend(
        handles=legend_handles,
        title="Effect Size\n$|\log_{2}\\frac{\mathrm{observed}}{\mathrm{shuffled}}|$",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
    )
    cax = inset_axes(
        axs[2],
        width="4%",
        height="50%",
        bbox_to_anchor=(1.12, -0.5, 1, 1),
        bbox_transform=axs[2].transAxes,
        loc="upper left",
        borderpad=0,
    )

    cbar = plt.colorbar(sc, cax=cax, orientation="vertical")
    cbar.set_label("Signed\n$log_{10}$ p-value", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    tick_locs = cbar.get_ticks()
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{int(abs(tick))}" for tick in tick_locs])


def heatmap_panel(ax, cbar_ax, data, font_size, title):
    sb.heatmap(
        data=data,
        cmap="Purples",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=True,
        cbar_ax=cbar_ax,
        cbar_kws={"ticks": [0.0, 1]},
        ax=ax,
    )

    ax.set_title(title, size=font_size * 1.1)
    ax.set_ylabel("VC area", size=font_size, rotation=0)
    ax.yaxis.set_label_coords(-0.05, 1.05)
    ax.set_xlabel("Coprojection target", size=font_size, labelpad=15)
    x_categories = data.columns
    tick_positions = np.arange(len(data.columns)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_categories, rotation=90, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size, rotation=0)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1)

    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.set_title(
        "P(target|VC area)",
        fontsize=font_size,
    )


def effect_size_panel(
    ax, cbar_ax, observed, shuffled, pval, size_scale, font_size, cmap="coolwarm"
):
    eff = np.log2(observed / shuffled)
    eff_long = (
        eff.stack(dropna=False)
        .rename("effect")
        .reset_index()
        .rename(columns={"level_0": "y", "level_1": "x"})
    )
    p_long = (
        pval.stack(dropna=False)
        .rename("p")
        .reset_index()
        .rename(columns={"level_0": "y", "level_1": "x"})
    )
    df = eff_long.merge(p_long, on=["y", "x"])

    df["bubble_size"] = df["effect"].abs() * size_scale
    df["color_val"] = np.sign(df["effect"]) * -np.log10(df["p"].clip(lower=1e-50))

    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    sc = ax.scatter(
        x=pd.Categorical(df["x"], categories=observed.columns, ordered=True).codes
        + 0.5,
        y=pd.Categorical(df["y"], categories=observed.index, ordered=True).codes,
        s=df["bubble_size"],
        c=df["color_val"],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )
    sig = df[df["p"] < 0.05]
    ax.scatter(
        x=pd.Categorical(sig["x"], categories=observed.columns).codes + 0.5,
        y=pd.Categorical(sig["y"], categories=observed.index).codes,
        s=sig["bubble_size"],
        facecolors="none",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xticks(np.arange(len(observed.columns)) + 0.5)
    ax.set_xticklabels(observed.columns, rotation=90, fontsize=font_size)
    ax.set_yticks(np.arange(len(observed.index)))
    ax.set_yticklabels(observed.index, fontsize=font_size)
    ax.invert_yaxis()
    ax.set_xlabel("Coprojection target", fontsize=font_size, labelpad=15)
    ax.yaxis.set_label_coords(-0.05, 1.05)
    ax.set_ylabel(
        "VC area",
        fontsize=font_size,
        rotation=0,
    )
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.ax.set_title("Signed\n$log_{10}$ p-value", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    handles = [
        ax.scatter([], [], s=v * size_scale, c="gray", alpha=0.5, label=v, linewidths=0)
        for v in (0.1, 0.5, 1.5)
    ]
    ax.legend(
        handles=handles,
        title="    Effect size\n| log$_{2}$$\\frac{\mathrm{observed}}{\mathrm{shuffled}}$ |",
        loc="upper left",
        bbox_to_anchor=(1, 0.3),
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
    )


def plot_the_heatmap_and_bubble(gen_parameters, all_mice_combined, fig):
    """function to plot conditional probability heatmap and effectsize in fig2"""
    font_size = gen_parameters["font_size"]
    cols_order = [
        "VISpl",
        "VISpor",
        "VISli",
        "VISl",
        "VISal",
        "VISrl",
        "VISa",
        "VISam",
        "VISpm",
        "VISp",
    ]

    obs, shuff, pmat = hf.prepare_conditional_probabilities(
        proj_path=gen_parameters["proj_path"],
        mouse_cfg=all_mice_combined,
        comp_VIS_only=True,
        cols_order=cols_order,
    )

    ax_hm = fig.add_axes([0.10, 0.25, 0.35, 0.60])
    cb_hm = fig.add_axes([0.46, 0.50, 0.01, 0.35])

    ax_eff = fig.add_axes([0.58, 0.25, 0.35, 0.60])
    cb_eff = fig.add_axes([0.94, 0.50, 0.01, 0.35])

    heatmap_panel(ax_hm, cb_hm, obs, font_size, "")
    effect_size_panel(
        ax_eff,
        cb_eff,
        observed=obs,
        shuffled=shuff,
        pval=pmat,
        size_scale=20,
        font_size=font_size,
    )
    fig.tight_layout()
    plt.show()


def add_sig_marker(x1, x2, y, p_val, gen_parameters, ax):
    if p_val < 0.05:
        ax.plot([x1, x1, x2, x2], [y, y, y, y], lw=0.5, color="black")
        ax.text(
            (x1 + x2) / 2,
            y + 0.02,
            f"p={p_val:.3f}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=gen_parameters["font_size"],
        )


def add_cosine_sim_plot(ax, cbax, cosine_df, gen_parameters):
    vmin = 0.9
    sb.heatmap(
        data=hf.convert_matrix_names(cosine_df),
        cmap="Purples",  #'RdBu_r',
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cbar_ax=cbax,
        vmin=vmin,
        vmax=1,
    )
    ax.tick_params(
        axis="y", which="major", labelsize=gen_parameters["font_size"], rotation=0
    )
    ax.tick_params(
        axis="x", which="major", labelsize=gen_parameters["font_size"], rotation=90
    )
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1)
    cbax.set_yticks([vmin, 1])
    cbax.tick_params(labelsize=gen_parameters["font_size"])
    cbax.set_title(
        "Cosine\nsimilarity", fontsize=gen_parameters["font_size"], loc="left"
    )


def add_motif_volcano_plot(gen_parameters, to_plot, ax):
    alpha_val_stream = 1
    to_plot["-log10_p_value"] = to_plot["-log10_p_value"].clip(upper=50)
    to_plot["stream"] = to_plot.index.map(hf.classify_stream)
    ax.tick_params(labelsize=gen_parameters["font_size"], width=0.25)
    for line in ["left", "bottom"]:
        ax.spines[line].set_linewidth(0.25)
    colors_to_colour, stream_labels = hf.get_stream_labels_and_colours()

    for stream in stream_labels.keys():
        plt.scatter(
            to_plot[to_plot["stream"] == stream]["shuf-sub"],
            to_plot[to_plot["stream"] == stream]["-log10_p_value"],
            color=colors_to_colour[stream],
            alpha=alpha_val_stream,
            s=5,
            edgecolor="white",
            linewidth=0.25,
            label=stream_labels[stream],
        )

    plt.axhline(y=-np.log10(0.05), color="black", linestyle="--", linewidth=0.5)
    convert_dict = hf.get_convert_dict()

    areas_to_check = [
        "VISrl, AUDv",
        "VISpl, VISli",
        "VISam, VISpor",
        "VISa, VISpm",
        "VISli, VISpor",
        "VISrl, VISli",
        "VISrl, VISpm",
        "VISa, ECT",
    ]
    for pair in areas_to_check:
        areas = pair.split(", ")
        x_val = to_plot.loc[pair, "shuf-sub"]
        y_val = to_plot.loc[pair, "-log10_p_value"]
        offset_y = y_val + 2
        if areas[0] in convert_dict:
            areas[0] = convert_dict[areas[0]]
        if areas[1] in convert_dict:
            areas[1] = convert_dict[areas[1]]
        offset_x = x_val + np.sign(x_val)
        plt.annotate(
            f"{areas[0]}, {areas[1]}",
            xy=(x_val, y_val),
            xytext=(offset_x, y_val),
            fontsize=gen_parameters["font_size"],
            ha="center",
            va="center",
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                linewidth=0.5,
                alpha=1,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0",
            ),
        )
    sb.despine(ax=ax, offset=5)
    plt.xlabel(
        "Effect size\nLog$_{2}$(observed/shuffled)",
        fontsize=gen_parameters["font_size"],
    )
    plt.ylabel("-Log$_{10}$(p-value)", fontsize=gen_parameters["font_size"])
    ax.tick_params(labelsize=gen_parameters["font_size"])
    plt.ylim(0, 55)
    plt.xlim(-2, 2)
    plt.legend(
        loc="upper left",
        prop={"size": 6},
        frameon=False,
        bbox_to_anchor=(-0.025, 1.25),
        handlelength=1,
    )


def plot_stream_effects(ax, to_plot, gen_parameters):
    colors_to_colour, stream_labels = hf.get_stream_labels_and_colours()
    to_plot["stream_label"] = to_plot["stream"].map(stream_labels)
    stream_order = ["Ventral-Ventral", "Dorsal-Dorsal", "Dorsal-Ventral"]
    plot_data = to_plot[to_plot["stream_label"].isin(stream_order)]
    # wrapped_labels = [textwrap.fill(label, width=8) for label in stream_order]
    dd = plot_data[plot_data["stream_label"] == "Dorsal-Dorsal"]["shuf-sub"]
    vv = plot_data[plot_data["stream_label"] == "Ventral-Ventral"]["shuf-sub"]
    dv = plot_data[plot_data["stream_label"] == "Dorsal-Ventral"]["shuf-sub"]

    stat_dd_dv, p_dd_dv = mannwhitneyu(dd, dv, alternative="two-sided")
    stat_vv_dv, p_vv_dv = mannwhitneyu(vv, dv, alternative="two-sided")
    stat_vv_dd, p_vv_dd = mannwhitneyu(vv, dd, alternative="two-sided")

    colors_dict = {
        "Dorsal-Dorsal": colors_to_colour["dorsal"],
        "Ventral-Ventral": colors_to_colour["ventral"],
        "Dorsal-Ventral": colors_to_colour["dorsal-ventral"],
    }

    for i, label in enumerate(stream_order):
        data_vals = plot_data[plot_data["stream_label"] == label]["shuf-sub"].values
        x_vals = np.random.normal(loc=i, scale=0.05, size=len(data_vals))  # jitter
        color = colors_dict[label]

        ax.scatter(
            x_vals,
            data_vals,
            s=20,
            facecolors="white",
            edgecolors=color,
            linewidths=0.5,
            alpha=0.3,
            zorder=1,
        )
        mean_val = np.mean(data_vals)
        ax.plot(
            [i - 0.25, i + 0.25],
            [mean_val, mean_val],
            linewidth=2,
            color=color,
            zorder=2,
        )

    sb.despine(ax=ax, offset=5)

    ax.set_xticks(range(len(stream_order)))
    ax.set_xticklabels(
        stream_order, rotation=45, fontsize=gen_parameters["font_size"], ha="right"
    )
    y_max = plot_data["shuf-sub"].max()
    bar_height = y_max + 0.2
    if p_dd_dv < 0.05:
        add_sig_marker(1, 2, bar_height, p_dd_dv, gen_parameters, ax)
        bar_height += 0.2
    if p_vv_dv < 0.05:
        add_sig_marker(0, 2, bar_height, p_vv_dv, gen_parameters, ax)
        bar_height += 0.2
    if p_vv_dd < 0.05:
        add_sig_marker(0, 1, bar_height, p_vv_dd, gen_parameters, ax)
        bar_height += 0.2
    ax.set_xlabel("")
    ax.set_ylabel(
        "Effect Size\n Log$_{2}$(observed/shuffled)",
        fontsize=gen_parameters["font_size"],
    )
    ax.tick_params(labelsize=gen_parameters["font_size"], width=0.25)
    for line in ["left", "bottom"]:
        ax.spines[line].set_linewidth(0.25)
    print(f"Dorsal-Dorsal vs Dorsal-Ventral: U={stat_dd_dv:.3f}, p={p_dd_dv:.4f}")
    print(f"Ventral-Ventral vs Dorsal-Ventral: U={stat_vv_dv:.3f}, p={p_vv_dv:.4f}")
    print(f"Ventral-Ventral vs Dorsal-Dorsal: U={stat_vv_dd:.3f}, p={p_vv_dd:.4f}")


def plot_corr_heatmap(gen_parameters):
    mice = gen_parameters["MICE"]
    area_dictionary = mdp.get_dict_area_vols(
        proj_path=gen_parameters["proj_path"], mice=gen_parameters["MICE"]
    )
    corr_dict = mdp.get_area_sample_corr(
        area_dictionary=area_dictionary, mice=gen_parameters["MICE"]
    )
    # order columns via similarity
    area_labels = corr_dict[mice[0]].index
    corr_stack = np.stack(
        [corr_dict[mouse].loc[area_labels, area_labels].values for mouse in mice],
        axis=0,
    )
    mean_corr = np.nanmean(corr_stack, axis=0)
    mean_df = pd.DataFrame(mean_corr, index=area_labels, columns=area_labels)
    distance_matrix = 1 - mean_df.values
    np.fill_diagonal(distance_matrix, 0)
    condensed = squareform(distance_matrix)
    linkage_matrix = linkage(condensed, method="ward")
    order = leaves_list(linkage_matrix)
    ordered_labels = mean_df.index[order]
    reordered = mean_df.loc[ordered_labels, ordered_labels]
    np.fill_diagonal(reordered.values, np.nan)
    # now plot
    g = sb.heatmap(
        data=hf.convert_matrix_names(reordered),
        cmap="coolwarm",
        center=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Spearman r"},
    )
    g.tick_params(
        axis="y", which="major", labelsize=gen_parameters["font_size"], rotation=0
    )

    for _, spine in g.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    cbar = g.collections[0].colorbar
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1)


def plot_simulation(gen_parameters, axs):
    eff_dict = hf.simulate_constant_vs_variable_labelling_efficiency()
    all_values = np.concatenate([df.values.flatten() for df in eff_dict.values()])
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)
    for i, title in enumerate(eff_dict.keys()):
        motif_df = eff_dict[title]
        sb.heatmap(
            ax=axs[i],
            data=motif_df,
            cmap="coolwarm",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=True,
            yticklabels=True,
        )
        axs[i].set_title(title, size=gen_parameters["font_size"] * 1.1)
        axs[i].tick_params(
            axis="y", which="major", labelsize=gen_parameters["font_size"], rotation=0
        )
        axs[i].tick_params(
            axis="x", which="major", labelsize=gen_parameters["font_size"]
        )
        axs[i].set_xlabel("Target area B")
        axs[i].set_ylabel("Target area A")
        for _, spine in axs[i].spines.items():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        cbar = axs[i].collections[0].colorbar
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(labelsize=gen_parameters["font_size"])
        cbar.set_label(
            "Log$_{2}$(observed/expected)", fontsize=gen_parameters["font_size"]
        )


def plot_observed_shuffle_approach_comparison(gen_parameters, all_mice_combined, axs):
    """supp fig plotting how effect size varies according to shuffle approach"""
    mice = gen_parameters["MICE"]
    proj_path = gen_parameters["proj_path"]
    observed_over_shuff = mdp.compare_shuffle_approaches(
        mice, proj_path, all_mice_combined
    )
    titles = ["Shuffled", "Curveball shuffled"]
    vis = gen_parameters["HVA_cols"]
    for title in observed_over_shuff.keys():
        new_mat = observed_over_shuff[title].loc[vis, vis]
        new_mat = hf.convert_matrix_names(new_mat)
        observed_over_shuff[title] = new_mat
    all_values = np.concatenate(
        [df.values.flatten() for df in observed_over_shuff.values()]
    )
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)
    for i, title in enumerate(observed_over_shuff.keys()):
        motif_df = observed_over_shuff[title]
        sb.heatmap(
            ax=axs[i],
            data=motif_df,
            cmap="coolwarm",
            center=0,
            vmin=vmin,
            vmax=vmax,
            xticklabels=True,
            yticklabels=True,
        )
        axs[i].set_title(titles[i], size=gen_parameters["font_size"] * 1.1)
        axs[i].tick_params(
            axis="y", which="major", labelsize=gen_parameters["font_size"], rotation=0
        )
        axs[i].tick_params(
            axis="x", which="major", labelsize=gen_parameters["font_size"], rotation=90
        )
        axs[i].set_xlabel("Target area B")
        axs[i].set_ylabel("Target area A")
        for _, spine in axs[i].spines.items():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        cbar = axs[i].collections[0].colorbar
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(labelsize=gen_parameters["font_size"])
        cbar.set_label(
            "Effect size\nLog$_{2}$(observed/shuffled)",
            fontsize=gen_parameters["font_size"],
        )

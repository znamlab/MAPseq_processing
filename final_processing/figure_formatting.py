import matplotlib

# from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import sys
from matplotlib.lines import Line2D
import colorsys

# rcParams['font.sans-serif'] = "Arial"
# rcParams['font.family'] = "Arial"


def set_font_params(gen_parameters):
    # matplotlib.rcParams['font.sans-serif'] = gen_parameters['font']
    rcParams["font.family"] = gen_parameters["font"]
    rcParams["font.size"] = gen_parameters["font_size"]
    rcParams["svg.fonttype"] = "none"
    rcParams["pdf.fonttype"] = 42
    rcParams["text.usetex"] = False


def print_sys():
    print(sys.executable, flush=True)


# def get_convert_dict():
#     convert_dict = {
#     "VISl": "LM",
#     "VISrl": "RL",
#     "VISal": "AL",
#     "VISa": "A",
#     "VISp": "V1",
#     "VISpor": "POR",
#     "VISli": "LI",
#     "VISpl": "P",
#     "VISpm": "PM",
#     "VISam": "AM"
# }
#     return convert_dict

# def get_colour_dict(allen_nomenclature=False):
#     convert_dict = {
#     "VISp": "V1",
#     "VISpor": "POR",
#     "VISli": "LI",
#     "VISal": "AL",
#     "VISl": "LM",
#     "VISpl": "P",
#     "VISpm": "PM",
#     "VISrl": "RL",
#     "VISam": "AM",
#     "VISa": "A"}
#     colour_dict = {'V1': '#4A443F',
#                         'POR': '#FF8623', #4C9E57
#                         'P': '#FFB845', #AAC255
#                         'LI': '#F5501E', #79B855
#                         'LM' : '#EB7C6C',
#                         'AL' : '#DB839F',
#                         'RL' : '#BB83DB',
#                         'A': '#8172F0',
#                         'AM': '#4062F0',
#                         'PM': '#6CBEFF',
#                         'OUT': 'lightgray',
#                         'ventral': '#F1842D',
#                         'dorsal': '#445FA9',
#                         'dorso-ventral': '#A161A4'} #FF8606
#     if allen_nomenclature:
#         new_colour_dict = {}
#         for vis_area in convert_dict:
#             new_colour_dict[vis_area] = colour_dict[convert_dict[vis_area]]
#     else:
#         new_colour_dict = colour_dict

#     return new_colour_dict


def myPlotSettings_splitAxis(fig, ax, ytitle, xtitle, title, axisColor="k", mySize=7):
    # mySize = 7 #18 for posters
    ax.spines["left"].set_color(axisColor)
    ax.spines["bottom"].set_color(axisColor)
    ax.xaxis.label.set_color(axisColor)
    ax.yaxis.label.set_color(axisColor)
    ax.tick_params(axis="x", colors=axisColor)
    ax.tick_params(axis="y", colors=axisColor)
    # plt.rcParams["font.family"] = myFont

    rcParams["font.size"] = mySize
    # ax.set_ylabel(ytitle)
    # ax.set_xlabel(xtitle)
    # ax.set_title(title,weight = 'bold')
    ax.set_ylabel(ytitle, fontsize=mySize, labelpad=1)
    ax.set_xlabel(xtitle, fontsize=mySize)
    ax.set_title(title, fontsize=mySize, weight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(mySize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(mySize)
    right = ax.spines["right"]
    right.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.25)
    for line in ["left", "bottom"]:
        ax.spines[line].set_linewidth(0.25)
        ax.spines[line].set_position(("outward", 3))
        # ax.spines['bottom'].set_position(('data', 7))


# def classify_stream(pair):
#     regions = pair.split(', ')
#     is_dorsal = [region in dorsal_stream for region in regions]
#     is_ventral = [region in ventral_stream for region in regions]

#     if all(is_dorsal):
#         return 'dorsal'
#     elif all(is_ventral):
#         return 'ventral'
#     elif any(is_dorsal) and any(is_ventral):
#         return 'dorsal-ventral'
#     else:
#         return 'other'

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
#     df_plot = dataframe.stack().reset_index()
#     df_plot.columns = ["y_label", "x_label", "conditional_prob"]
#     df_plot["p_value"] = pval_df.stack().reset_index(drop=True)
#     x_categories = dataframe.columns
#     y_categories = dataframe.index

#     df_plot["x"] = pd.Categorical(df_plot["x_label"], categories=x_categories, ordered=True).codes
#     df_plot["y"] = pd.Categorical(df_plot["y_label"], categories=y_categories, ordered=True).codes
#     df_plot["bubble_size"] = df_plot["conditional_prob"].abs() * size_scale
#     df_plot["color_value"] = -np.log10(df_plot["p_value"].clip(lower=1e-50))
#     purples_cmap = plt.cm.Purples
#     new_colors = purples_cmap(np.linspace(0.15, 1, 256))
#     custom_purples = mcolors.LinearSegmentedColormap.from_list("custom_purples", new_colors)

#     fig, ax = plt.subplots(figsize=(subplot_size[0], subplot_size[1]))
#     sc = ax.scatter(
#         x=df_plot["x"],
#         y=df_plot["y"],
#         s=df_plot["bubble_size"],
#         c=df_plot["color_value"],
#         cmap=custom_purples,
#         vmin=df_plot["color_value"].min() * 0.8,
#         vmax=df_plot["color_value"].max(),
#         edgecolors="none"
#     )
#     df_plot["significant"] = df_plot["p_value"] < 0.05
#     df_signif = df_plot[df_plot["significant"]]

#     ax.scatter(
#         x=df_signif["x"],
#         y=df_signif["y"],
#         s=df_signif["bubble_size"],
#         facecolors="none",
#         edgecolors="black",
#         linewidths=0.5
#     )


#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label("-log10(p-value)", fontsize=5)
#     cbar.ax.tick_params(labelsize=5)
#     legend_values = [0.1, 0.4, 0.8]
#     legend_handles = [
#         ax.scatter([], [], s=val * size_scale, c="gray", alpha=0.5,
#                    label=fr"$\mathit{{P(target \mid VC\ area)}}$ = {val}")
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
#         borderaxespad=0.,
#         frameon=True,
#         handleheight=2.0,
#         fontsize=font_size, title_fontsize=font_size
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

# def combine_broad_regions(dataframe, regions_to_add):
#     summed_data = {}
#     for area, tubes in regions_to_add.items():
#         valid_tubes = [tube for tube in tubes if tube in dataframe.columns]
#         summed_data[area] = dataframe[valid_tubes].sum(axis=1)

#     df_result = pd.DataFrame(summed_data)
#     df_result = df_result.loc[(df_result != 0).any(axis=1)]
#     return df_result


def convert_to_exp(num, sig_fig=2):
    mant, exp = f"{num:.{sig_fig}E}".split("E")
    if round(float(mant)) == 1:
        exp_add = exp
        symbol_add = "="
    else:
        exp_add = int(exp) + 1
        symbol_add = "<"
    return symbol_add, exp_add


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


# def plot_exponential_fit(ax, fit: FitResult):
#     ax.plot(fit.fitted_x, fit.fitted_y, color='black',
#             lw=0.5, ls='dotted', label='Exponential fit\nℓ = '
#             f'{1/fit.params[1]:.0f} µm')


def style_figure_exp_fits(fig, axes, color_map, convert_dict, font_size):
    axes[0].set_ylabel("Projection probability", fontsize=font_size)
    axes[1].set_ylabel("Normalised projection density", fontsize=font_size)
    for ax in axes:
        ax.set_xticks([0, 2000, 4000])

    labels = [convert_dict.get(a, a) for a in color_map]
    dummy = [Line2D([], [], ls="none") for _ in labels]
    legend = fig.legend(
        dummy,
        labels,
        loc="center left",
        bbox_to_anchor=(0.4, 0.5),
        frameon=False,
        handlelength=0,
        handletextpad=0.1,
        fontsize=font_size,
        prop={"family": "Arial"},
    )
    for txt, area in zip(legend.get_texts(), color_map):
        txt.set_color(color_map[area])


def adjust_color(color, amount=1.0):
    """function to change the change the lightness of a specified color in the plot"""
    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    r, g, b = matplotlib.colors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(min(l * amount, 1.0), 0.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


# def plot_ap_vs_visap(ax, data, gen_parameters, ff):
#     """
#     function to draw AP-Vis-to-soma scatter plot and regression on supplied ax and return rho and p-value.
#     """
#     convert_dict = ff.get_convert_dict()
#     data = data.copy()
#     data["converted"] = data["VC_majority"].map(convert_dict)
#     palette = ff.get_colour_dict(allen_nomenclature=False)
#     rho, pval = pearsonr(data["AP_Vis"], data["mean_AP_soma"])
#     sb.regplot(
#         x="AP_Vis",
#         y="mean_AP_soma",
#         data=data,
#         color="black",
#         scatter=False,
#         ci=95,
#         scatter_kws={"s": 5},
#         line_kws={"linewidth": 1},
#         ax=ax,
#     )
#     scatter = sb.scatterplot(
#         x="AP_Vis",
#         y="mean_AP_soma",
#         data=data,
#         hue="converted",
#         legend=False,
#         palette=palette,
#         s=5,
#         ax=ax,
#     )
#     symbol_add, exp_add = ff.convert_to_exp(num=pval)
#     ax.text(
#         0.7,
#         0.25,
#         rf"$r={rho:.3f}$" + f"\n$p {symbol_add} 10^{{{exp_add}}}$",
#         ha="left",
#         va="top",
#         transform=ax.transAxes,
#         fontsize=gen_parameters["font_size"],
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
#     )
#     xlabel = "Cubelet A-P\nposition (µm)"
#     ylabel = "Mean soma A-P\nposition (µm)"
#     ax.text(
#         1.45,
#         -0.3,
#         "Anterior",
#         ha="right",
#         va="bottom",
#         transform=ax.transAxes,
#         fontsize=gen_parameters["font_size"] * 0.9,
#     )
#     ax.text(
#         -0.4,
#         1.1,
#         "Anterior",
#         ha="center",
#         va="bottom",
#         transform=ax.transAxes,
#         fontsize=gen_parameters["font_size"] * 0.9,
#     )
#     ax.text(
#         -0.4,
#         -0.3,
#         "Posterior",
#         transform=ax.transAxes,
#         fontsize=gen_parameters["font_size"] * 0.9,
#         va="bottom",
#         ha="center",
#     )
#     myPlotSettings_splitAxis(
#         fig=ax.figure,
#         ax=ax,
#         ytitle=ylabel,
#         xtitle=xlabel,
#         title="",
#         mySize=gen_parameters["font_size"],
#     )
#     ax.set_xticks([0, 3000])

#     return rho, pval

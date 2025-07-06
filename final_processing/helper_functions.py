import pandas as pd
import final_processing.MAPseq_data_processing as mdp
import final_processing.loading_functions as lf
import numpy as np


def get_convert_dict():
    convert_dict = {
        "VISl": "LM",
        "VISrl": "RL",
        "VISal": "AL",
        "VISa": "A",
        "VISp": "V1",
        "VISpor": "POR",
        "VISli": "LI",
        "VISpl": "P",
        "VISpm": "PM",
        "VISam": "AM",
    }
    return convert_dict


def get_colour_dict(allen_nomenclature=False):
    convert_dict = {
        "VISp": "V1",
        "VISpor": "POR",
        "VISli": "LI",
        "VISal": "AL",
        "VISl": "LM",
        "VISpl": "P",
        "VISpm": "PM",
        "VISrl": "RL",
        "VISam": "AM",
        "VISa": "A",
    }
    colour_dict = {
        "V1": "#4A443F",
        "POR": "#FF8623",  # 4C9E57
        "P": "#FFB845",  # AAC255
        "LI": "#F5501E",  # 79B855
        "LM": "#EB7C6C",
        "AL": "#DB839F",
        "RL": "#BB83DB",
        "A": "#8172F0",
        "AM": "#4062F0",
        "PM": "#6CBEFF",
        "OUT": "lightgray",
        "ventral": "#F1842D",
        "dorsal": "#445FA9",
        "dorso-ventral": "#A161A4",
    }  # FF8606
    if allen_nomenclature:
        new_colour_dict = {}
        for vis_area in convert_dict:
            new_colour_dict[vis_area] = colour_dict[convert_dict[vis_area]]
    else:
        new_colour_dict = colour_dict

    return new_colour_dict


def classify_stream(pair):
    dorsal_stream, ventral_stream, visual = lf.load_dorsal_ventral_definitions()
    regions = pair.split(", ")
    is_dorsal = [region in dorsal_stream for region in regions]
    is_ventral = [region in ventral_stream for region in regions]
    is_visual = [region in visual for region in regions]
    is_visp = [region == "VISp" for region in regions]

    if all(is_dorsal):
        return "dorsal"
    elif all(is_ventral):
        return "ventral"
    elif any(is_dorsal) and any(is_ventral):
        return "dorsal-ventral"
    elif any(is_visp) and all(is_visual):
        return "visp-hva"
    elif any(is_visual) and not all(is_visual):
        return "visual-nonvisual"
    else:
        return "other"


def combine_broad_regions(dataframe, regions_to_add):
    summed_data = {}
    for area, tubes in regions_to_add.items():
        valid_tubes = [tube for tube in tubes if tube in dataframe.columns]
        summed_data[area] = dataframe[valid_tubes].sum(axis=1)

    df_result = pd.DataFrame(summed_data)
    df_result = df_result.loc[(df_result != 0).any(axis=1)]
    return df_result


def convert_matrix_names(matrix):
    convert_dict = {
        "VISp": "V1",
        "VISpor": "POR",
        "VISli": "LI",
        "VISal": "AL",
        "VISl": "LM",
        "VISpl": "P",
        "VISpm": "PM",
        "VISrl": "RL",
        "VISam": "AM",
        "VISa": "A",
    }
    matrix.rename(columns=convert_dict, inplace=True)
    matrix.rename(index=convert_dict, inplace=True)
    return matrix


def prepare_conditional_probabilities(proj_path, mouse_cfg, comp_VIS_only, cols_order):
    pval_adj, cp_dict = mdp.get_p_val_comp_to_shuffled(
        proj_path=proj_path, all_mice_combined=mouse_cfg, comp_VIS_only=comp_VIS_only
    )
    # corr   = cp_dict["observed"].corr()
    # order  = dendrogram(linkage(corr, method="ward"),
    #                     labels=corr.columns,
    #                     no_plot=True)["ivl"]
    order = [
        "AUDv",
        "AUDpo",
        "AUDd",
        "TEa",
        "ECT",
        "PERI",
        "SSs",
        "SSp",
        "MOp",
        "MOs",
        "RSPagl",
        "RSPd",
        "ACAd",
        "RSPv",
        "ACAv",
    ]
    order = cols_order + [c for c in order if c not in cols_order]

    obs = cp_dict["observed"].loc[cols_order][order]
    shuff = cp_dict["shuffled"].loc[cols_order][order]
    pval = pval_adj.loc[cols_order][order]

    conv_obs = convert_matrix_names(obs)
    conv_shuff = convert_matrix_names(shuff)
    conv_pval = convert_matrix_names(pval)

    return conv_obs, conv_shuff, conv_pval


def get_stream_labels_and_colours():
    colors_to_colour = {
        "visual-nonvisual": "#A1A1A1",
        "ventral": "#F1842D",
        "dorsal": "#445FA9",
        "dorsal-ventral": "#A161A4",
        "visp-hva": "black",
    }
    stream_labels = {
        "visual-nonvisual": "Visual-Other",
        "visp-hva": "V1-HVA",
        "dorsal": "Dorsal-Dorsal",
        "ventral": "Ventral-Ventral",
        "dorsal-ventral": "Dorsal-Ventral",
    }
    return colors_to_colour, stream_labels


def simulate_constant_vs_variable_labelling_efficiency():
    n_areas = 10
    proj_prob = np.ones(n_areas) * 0.2
    titles = ["Uniform efficiency", "Variable efficiency"]
    n_neurons = 100000
    efficiency = np.random.rand(n_neurons)
    constant_efficiency = np.ones(n_neurons) * 0.5
    eff_dict = {}
    for i, which_eff in enumerate([constant_efficiency, efficiency]):
        neurons_proj_prob = proj_prob[None, :] * which_eff[:, None]
        neurons_proj = (
            np.random.rand(n_neurons, n_areas) < neurons_proj_prob
        )  # randomly simulate the detection of each neuron–area projection based on the projection probability and efficiency. take a random value between 0 and 1 for each area for each barcode. If the random value is less than the effiicency adjusted projection probability, we observe it as projecting there
        motif_df = pd.DataFrame(np.zeros((n_areas, n_areas)))
        for area_a in range(n_areas):
            for area_b in range(n_areas):
                if area_a == area_b:
                    continue
                observed = (
                    np.count_nonzero(neurons_proj[:, area_a] & neurons_proj[:, area_b])
                ) / n_neurons
                expected = ((np.count_nonzero(neurons_proj[:, area_a])) / n_neurons) * (
                    (np.count_nonzero(neurons_proj[:, area_b])) / n_neurons
                )
                motif_df.iloc[area_a, area_b] = np.log2(observed / expected)
        np.fill_diagonal(motif_df.values, np.nan)
        eff_dict[titles[i]] = motif_df
    return eff_dict

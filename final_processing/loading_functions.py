# import matplotlib
import yaml
from pathlib import Path
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def load_gen_params():
    with open("general_analysis_parameters.yaml", "r") as file:
        gen_parameters = yaml.safe_load(file)
        return gen_parameters


def flatten_dict(d):
    flattened_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened_dict[subkey] = subvalue
        else:
            flattened_dict[key] = value
    return flattened_dict


def load_parameters(directory):
    """Load the parameters yaml file containing all the parameters required for
    preprocessing MAPseq data

    Args:
    directory (str): Directory where to load parameters from. Default 'root' for the
        default parameters (found in `mapseq_preprocessing/parameters.py`).

    Returns:
        dict: contents of parameters.yml
    """

    if directory == "root":
        parameters_file = Path(__file__).parent / "parameters.yml"
    else:
        parameters_file = Path(directory) / "parameters.yml"
    with open(parameters_file, "r") as f:
        parameters = flatten_dict(yaml.safe_load(f))
    return parameters


def load_shuffled_matrices(proj_path):
    """we've previously performed shuffling using the curveball algorithm, recorded co-projections in shuffled and collated the dataset."""
    shuffled_numbers = pd.read_pickle(
        f"{proj_path}/collated_shuffles/shuffled__neuron_numbers_cubelet__collated.pkl"
    )
    shuffled_2_combinations = pd.read_pickle(
        f"{proj_path}/collated_shuffles/shuffled_cubelet_2_comb__collated.pkl"
    )
    shuffle_total_numbers = pd.read_pickle(
        f"{proj_path}/collated_shuffles/total_neuron_numbers_cubelet__collated.pkl"
    )
    return shuffled_numbers, shuffled_2_combinations, shuffle_total_numbers


def load_dorsal_ventral_definitions():
    dorsal_stream = ["VISa", "VISam", "VISpm", "VISrl"]
    ventral_stream = ["VISpor", "VISpl", "VISl", "VISli", "VISal"]
    visual = [
        "VISa",
        "VISam",
        "VISpm",
        "VISrl",
        "VISpor",
        "VISpl",
        "VISl",
        "VISli",
        "VISal",
        "VISp",
    ]
    return dorsal_stream, ventral_stream, visual


def load_allen_anterograde():
    """we selected wt mice anterograde tracing with at least 75% in A1"""
    mcc = MouseConnectivityCache(resolution=10)
    experiment_id_a = 120491896  # AUDp
    experiment_id_b = 116903230  # AUDp, AUDpo, AUDd, AUDv
    experiment_id_c = 100149109  # AUDp and AUDd
    expt_a, pd_a_info = mcc.get_projection_density(experiment_id_a)
    expt_b, pd_b_info = mcc.get_projection_density(experiment_id_b)
    expt_c, pd_c_info = mcc.get_projection_density(experiment_id_c)
    expts = [expt_a, expt_b, expt_c]
    return expts

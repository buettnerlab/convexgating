#!/usr/bin/env python
import os
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def do_SCATTER(
    clust_string,
    general,
    general_targ,
    general_non_targ,
    hierarch,
    re_gating_dict,
    hull_dict,
    gate_points_dict,
    key,
    show_SCATTER=True,
    save_SCATTER=True,
    save_path=os.getcwd(),
):
    """

    Visualization of gating strategy via scatterplots.

    Parameters
    ----------
    clust_string : string
        an identifier for the current cluster, e.g. '4'
    general : pd.DataFrame
        dataframe containing all relevant infos for visualization
        output of function 'process_results'
    general_targ : pd.DataFrame
        dataframe containing all relevant infos for visualization for target population
    general_non_targ : pd.DataFrame
        dataframe containing all relevant infos for visualization for non_target population
    hierarch : int
        current hierarchy
    re_gating_dict : dict
        output of function 'process_results'
    hull_dict : dict
        output of function 'apply_convex_hull'
    gate_points_dict : dict
        output of function 'apply_convex_hull'
    key : int
        internal identifier
    show_SCATTER : True or False (default True)
        whether to print scatter plot on console
    save_SCATTER : True or False (default True)
        whether to save scatter plot
    save_path : str (default os.getcwd() -> current working directory)
        path (location) to save graphic

    Returns
    -------
    TYPE
        DESCRIPTION.
    xlim : array
        lower and upper bound on x-axis (first marker)
    ylim : array
        lower and upper bound on y-axis (second marker)
    """

    gen_targ_h = general_targ[general_targ["final_gate_" + str(hierarch - 1)] == 1]
    gen_non_targ_h = general_non_targ[general_non_targ["final_gate_" + str(hierarch - 1)] == 1]
    markers = list(re_gating_dict[key][0][str(hierarch)].columns[0:2].values)
    plt.figure()
    plt.plot(
        gen_non_targ_h[markers].values[:, 0],
        gen_non_targ_h[markers].values[:, 1],
        "o",
        color="dodgerblue",
        markersize=3,
    )
    plt.plot(gen_targ_h[markers].values[:, 0], gen_targ_h[markers].values[:, 1], "o", color="darkorange", markersize=3)
    plt.xlabel(markers[0], fontsize=14)
    plt.ylabel(markers[1], fontsize=14)
    plt.title("Hierarchy " + str(hierarch) + " - Scatter", fontsize=15)
    for simplex in hull_dict[hierarch].simplices:
        plt.plot(gate_points_dict[hierarch][simplex, 0], gate_points_dict[hierarch][simplex, 1], "r-", linewidth=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    if save_SCATTER:
        path_out = os.path.join(save_path, "cluster_" + clust_string)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        save_location = os.path.join(path_out, "scatter_h" + str(hierarch))
        plt.savefig(save_location, bbox_inches="tight")

    if show_SCATTER:
        plt.show()
    else:
        plt.close()
    return xlim, ylim


def do_HEAT_targets(
    clust_string,
    general,
    general_targ,
    hierarch,
    re_gating_dict,
    hull_dict,
    gate_points_dict,
    xlim,
    ylim,
    key,
    show_HEAT=True,
    save_HEAT=True,
    save_path=os.getcwd(),
):
    """
    Visualization of gating strategy via heatmaps (target population).

    Parameters
    ----------
    clust_string : string
        an identifier for the current cluster, e.g. '4'
    general : pd.DataFrame
        dataframe containing all relevant infos for visualization
        output of function 'process_results'
    general_targ : pd.DataFrame
        dataframe containing all relevant infos for visualization for target population
    hierarch : int
        current hierarchy
    re_gating_dict : dict
        output of function 'process_results'
    hull_dict : dict
        output of function 'apply_convex_hull'
    gate_points_dict : dict
        output of function 'apply_convex_hull'
    xlim : array
        lower and upper bound on x-axis (first marker) -> output of function 'do_SCATTER'
    ylim : array
        lower and upper bound on y-axis (second marker) -> output of function 'do_SCATTER'
    key : int
        internal identifier
    show_HEAT : True or False (default True)
        whether to print heat plot for target population on console
    save_HEAT : True or False (default True)
        whether to save heat plot for target population
    save_path : str (default os.getcwd() -> current working directory)
        path (location) to save graphic

    """
    gen_targ_h = general_targ[general_targ["final_gate_" + str(hierarch - 1)] == 1]
    markers = list(re_gating_dict[key][0][str(hierarch)].columns[0:2].values)
    plt.figure()
    plt.hexbin(
        gen_targ_h[markers].values[:, 0],
        gen_targ_h[markers].values[:, 1],
        gridsize=(70, 70),
        cmap="inferno",
        extent=xlim + ylim,
    )
    plt.xlabel(markers[0], fontsize=14)
    plt.ylabel(markers[1], fontsize=14)
    plt.title("Hierarchy " + str(hierarch) + " - Density Targets", fontsize=15)
    for simplex in hull_dict[hierarch].simplices:
        plt.plot(gate_points_dict[hierarch][simplex, 0], gate_points_dict[hierarch][simplex, 1], "r-", linewidth=3)
    if save_HEAT:
        path_out = os.path.join(save_path, "cluster_" + clust_string)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        save_location = os.path.join(path_out, "heat_targets_h" + str(hierarch))
        plt.savefig(save_location, bbox_inches="tight")
    if show_HEAT:
        plt.show()
    else:
        plt.close()


def do_HEAT_non_targets(
    clust_string,
    general,
    general_non_targ,
    hierarch,
    re_gating_dict,
    hull_dict,
    gate_points_dict,
    xlim,
    ylim,
    key,
    show_HEAT=True,
    save_HEAT=True,
    save_path=os.getcwd(),
):
    """
    Visualization of gating strategy via heatmaps (non-target population).

    Parameters
    ----------
    clust_string : string
        an identifier for the current cluster, e.g. '4'
    general : pd.DataFrame
        dataframe containing all relevant infos for visualization
        output of function 'process_results'
    general_non_targ : pd.DataFrame
        dataframe containing all relevant infos for visualization for non_target population
    hierarch : int
        current hierarchy
    re_gating_dict : dict
        output of function 'process_results'
    hull_dict : dict
        output of function 'apply_convex_hull'
    gate_points_dict : dict
        output of function 'apply_convex_hull'
    xlim : array
        lower and upper bound on x-axis (first marker) -> output of function 'do_SCATTER'
    ylim : array
        lower and upper bound on y-axis (second marker) -> output of function 'do_SCATTER'
    key : int
        internal identifier
    show_HEAT : True or False (default True)
        whether to print heat plot for non_target population on console
    save_HEAT : True or False (default True)
        whether to save heat plot for non_target population
    save_path : str (default os.getcwd() -> current working directory)
        path (location) to save graphic
    """
    gen_non_targ_h = general_non_targ[general_non_targ["final_gate_" + str(hierarch - 1)] == 1]
    markers = list(re_gating_dict[key][0][str(hierarch)].columns[0:2].values)
    plt.hexbin(
        gen_non_targ_h[markers].values[:, 0],
        gen_non_targ_h[markers].values[:, 1],
        gridsize=(70, 70),
        cmap="inferno",
        extent=xlim + ylim,
    )
    plt.xlabel(markers[0], fontsize=14)
    plt.ylabel(markers[1], fontsize=14)
    plt.title("Hierarchy " + str(hierarch) + " - Density Non Targets", fontsize=15)
    for simplex in hull_dict[hierarch].simplices:
        plt.plot(gate_points_dict[hierarch][simplex, 0], gate_points_dict[hierarch][simplex, 1], "r-", linewidth=3)
    if save_HEAT:
        path_out = os.path.join(save_path, "cluster_" + clust_string)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        save_location = os.path.join(path_out, "heat_non_targets_h" + str(hierarch))
        plt.savefig(save_location, bbox_inches="tight")
    if show_HEAT:
        plt.show()
    else:
        plt.close()


def do_plot_metrics(
    clust_string, scores, key, overview, save_metrics_plot=True, show_metrics_plot=True, save_path=os.getcwd()
):
    """
    Graphical performance overview.

    Parameters
    ----------
    clust_string : string
        an identifier for the current cluster, e.g. '4'
    scores : pd.DataFrame
        overview of gating performance -> output of function 'metrics_new'
    key : int
        internal identifier
    overview : pd.DataFrame
        output of function 'do_complete_gating'
    show_metrics_plot : True or False (default True)
        whether to print performance graphic on console
    save_metrics_plot : True or False (default True)
        hether to save performance graphic
    save_path : str (default os.getcwd() -> current working directory)
        path (location) to save graphic
    """
    f1_val = scores.T["f1"]
    recall_val = scores.T["recall"]
    precision_val = scores.T["precision"]
    y_names = list(scores.T.index)
    plt.figure()
    plt.plot(y_names, recall_val, "-o", color="indigo", label="recall", linestyle="dashdot")
    plt.plot(y_names, f1_val, "-o", color="darkgreen", label="F1", linestyle="dashdot")
    plt.plot(y_names, precision_val, "-o", color="darkgoldenrod", label="precision", linestyle="dashdot")
    plt.legend()
    plt.title("Scores - Cluster " + overview[overview["key"] == key]["cluster_number"].values[0], fontsize=15)
    plt.ylim([0, 1.01])
    plt.ylabel("Measure", fontsize=14)
    plt.xlabel("Gating Depth", fontsize=14)

    if save_metrics_plot:
        path_out = os.path.join(save_path, "cluster_" + clust_string)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        save_location = os.path.join(path_out, "performance_graphic.png")
        plt.savefig(save_location)
    if show_metrics_plot:
        plt.show()
    else:
        plt.close()

#!/usr/bin/env python
import os

# import time
import warnings

import numpy as np
import pandas as pd
import sklearn

from .helper import (
    GradientDescentMulti,
    apply_convex_hull,
    classify_points,
    create_target_df,
    fraction_targets_vanilla,
    generate2unit_planes,
    get_candidate_points,
    get_new_marker_combo,
    get_normal_v_biases,
    get_points_to_connect,
    get_relevant_points,
    get_two_points_each_hyperplane,
    initialize_norm_bias,
    metrics_new,
    normalization,
    only_f1,
    preprocess_adata_gating,
    process_results,
    return_best_marker_combo,
    return_best_marker_combo_single_svm,
    return_best_marker_combo_single_tree,
    add_tight_hull_hierarchy,
    add_gate_tight,
    add_tight_metric,
    add_visualization_hierarchy,
    add_tight_analysis,
    plot_metric_tight,
    get_f1_hierarch,
    make_performance_summary,
    make_marker_summary,
    add_gating_to_anndata,
    add_gate_locations_to_anndata,
    add_performance_to_anndata,
    add_marker_summary_to_anndata,
    add_performance_summary_to_anndata,
    updata_anndata_uns
)
from .hyperparameters import (
    PC,
    arange_init,
    batch_size,
    grid_divisor,
    iterations,
    learning_rate,
    marker_sel,
    nr_hyperplanes,
    nr_max_hierarchies,
    refinement_grid_search,
    save_HEAT,
    save_metrics_df,
    save_metrics_plot,
    save_SCATTER,
    show_HEAT,
    show_metrics_df,
    show_metrics_plot,
    show_SCATTER,
    weight_version,
)
from .plotting import do_HEAT_non_targets, do_HEAT_targets, do_plot_metrics, do_SCATTER

warnings.filterwarnings("ignore")

def CONVEX_GATING(adata,cluster_numbers,cluster_string,save_path=os.getcwd(), add_noise=True, update_anndata=True,focus="f1"):
    """
    Derives gating strategies for selected clusters.

    Parameters
    ----------
    adata : object
        AnnData object, with label information in `adata.obs`.
    cluster_numbers : list
        List containing the cluster numbers or names to derive gating strategies for.
    cluster_string : str
        Column name in `adata.obs` with label or cluster information corresponding to `cluster_numbers`.
    save_path : str, optional
        Path to folder where gating output will be saved. Creates folder if it doesn't exist. Default is the current working directory.
    add_noise : bool, optional
        Indicates whether a small amount of random noise is added for internal stability. Default is `True`.
    update_anndata : bool, optional
        Indicates whether gating output is saved in `adata.uns` in addition to the output folder. Default is `True`.
    focus : str, optional
        Specifies whether CG focuses on high "f1" (default) or "recall".

    Returns
    -------
    adata : object
        AnnData object containing gating information if `update_anndata` is `True`.

    Notes
    -----
    Gating output is saved in the location specified by the `save_path` parameter.
    
    """
    
    gating_strategy(adata = adata,
                         cluster_numbers = cluster_numbers,
                         cluster_string = cluster_string,
                         save_path=save_path,
                             add_noise = add_noise,focus=focus)
    convex_hull_add_on(meta_info_path = os.path.join(save_path, 'meta_info.npy'),target_location=save_path)
    if update_anndata:
        return updata_anndata_uns(adata,save_path)

def gating_strategy(adata, cluster_numbers, cluster_string, save_path=os.getcwd(), add_noise=True,focus="f1"):
    """
    Learning gating strategy for specific cell clusters.

    Parameters
    ----------
    adata : AnnData object
        complete input Data for gating.
    cluster_numbers : list of strings or ints or doubles
        list of cluster numbers/identifieres in adata.obs[cluster_string] to find gating strategy for
         -> e.g ['4','5'] to find gating strategies for cell population '4' and '5'
    cluster_string : string
        column name in adata.obs where cluster labels are found, e.g 'louvain'
    add_noise : True or False
        if True add small amount of nice to count data for more stable internal procedures
    focus: "f1" or "recall" 
        default focus "f1", optional "recall"
    """
    if add_noise:
        adata.X = adata.X + (np.random.rand(adata.X.shape[0], adata.X.shape[1]) - 0.5) / 10000

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cell_data = preprocess_adata_gating(adata, cluster_string)
    channels = list(adata.var.index)
    # print('Checkpoint 0')
    clust_string_dict, re_gating_dict, general_dict = FIND_GATING_STRATEGY(
        cell_data,
        channels,
        cluster_numbers,
        nr_max_hierarchies=nr_max_hierarchies,
        PC=PC,
        cluster_string=cluster_string,
        learning_rate=learning_rate,
        iterations=iterations,
        batch_size=batch_size,
        nr_hyperplanes=nr_hyperplanes,
        grid_divisor=grid_divisor,
        refinement_grid_search=refinement_grid_search,
        weight_version=weight_version,
        marker_sel=marker_sel,
        arange_init=arange_init,
        show_HEAT=show_HEAT,
        save_HEAT=save_HEAT,
        show_SCATTER=show_SCATTER,
        save_SCATTER=save_SCATTER,
        show_metrics_df=show_metrics_df,
        save_metrics_df=save_metrics_df,
        save_metrics_plot=save_metrics_plot,
        show_metrics_plot=show_metrics_plot,
        save_path=save_path,
        focus=focus,
    )
    # print('Checkpoint 7')
    meta_info = {}
    meta_info["clusterkeys"] = clust_string_dict
    meta_info["gating_summary"] = re_gating_dict
    meta_info["general_summary"] = general_dict
    np.save(os.path.join(save_path, "meta_info.npy"), meta_info)
    # np.load('meta_info.npy',allow_pickle='TRUE').item() to read again


def FIND_GATING_STRATEGY(
    cell_data,
    channels,
    cluster_numbers,
    first_hierarchy=1,
    nr_max_hierarchies=5,
    cluster_string="louvain",
    PC=True,
    visualize_gate=True,
    learning_rate=0.05,
    iterations=50,
    nr_hyperplanes=8,
    batch_size=5000,
    grid_divisor=4,
    arange_init=[0, 2, 4, 8, 10],
    refinement_grid_search=4,
    weight_version=1,
    show_everything=False,
    max_try=5,
    marker_sel="heuristic",
    show_HEAT=True,
    save_HEAT=True,
    show_SCATTER=True,
    save_SCATTER=True,
    show_metrics_df=True,
    save_metrics_df=True,
    save_metrics_plot=True,
    show_metrics_plot=True,
    save_path=os.getcwd(),
    focus = "f1",
):
    """
    Finding a gating strategy.

    Parameters
    ----------
    cell_data : pd.DataFrame
        size (n_cells,n_markers) -> output of preprocess_adata_gating
    channels : list
        list of strings to indicate which channels to use for gating
    cluster_numbers : list of strings
        list of clusters to find gating strategy (e.g. ['2','4'])
    first_hierarchy : int
        start hierarch. The default is 1.
    nr_max_hierarchies : int
        maximal hierarchies in gating strategy. The default is 5.
    cluster_string : string
        name of column in cell_data that contains labels. The default is 'louvain'.
    PC : TYPE, optional
        DESCRIPTION. The default is True.
    visualize_gate : TYPE, optional
        DESCRIPTION. The default is False.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.05.
    iterations : TYPE, optional
        DESCRIPTION. The default is 50.
    nr_hyperplanes : TYPE, optional
        DESCRIPTION. The default is 8.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 5000.
    grid_divisor : TYPE, optional
        DESCRIPTION. The default is 2.
    arange_init : TYPE, optional
        DESCRIPTION. The default is [0,2,4,8,10].
    refinement_grid_search : TYPE, optional
        DESCRIPTION. The default is 2.
    weight_version : int, optional
        0,1,2 -> if 0 : negative proportional
              -> if 1 : negative proportional only if more non-targets than targets -> otherwise equal weight
              -> if 2 : always equal weight. The default is 1.
    marker_sel : TYPE, optional
        DESCRIPTION. The default is 'heuristic'.
    show_everything : boolean, optional
        True if gating procedure should be visualized, False if not. The default is True.
    focus: "f1" or "recall" -> default focus "f1", optional "recall"

    Returns
    -------
    TYPE
        DESCRIPTION.
    best_hierarchy : list
        list of hierarchies for an optimal gating panel.
    summary_gating_dicts : dict
        summary of gating results per hierarchy.
    summary_losses_dicts : dict
        summary of the loss function results per hierarchy.
    target_df_dicts : dict of pandas DataFrames
        indicating whether a cell is part of a gate or not.
    renorm_df : TYPE
        DESCRIPTION.
    in_gate_dicts : TYPE
        DESCRIPTION.

    """
    # print('Checkpoint 1')
    overview, results_dictionaries, renorm_df_dict, res_in_gates = do_complete_gating(
        cell_data,
        channels,
        cluster_numbers,
        nr_max_hierarchies=nr_max_hierarchies,
        cluster_string=cluster_string,
        PC=PC,
        visualize_gate=False,
        learning_rate=learning_rate,
        iterations=iterations,
        nr_hyperplanes=nr_hyperplanes,
        batch_size=batch_size,
        grid_divisor=grid_divisor,
        arange_init=arange_init,
        refinement_grid_search=refinement_grid_search,
        weight_version=weight_version,
        show_everything=False,
        max_try=5,
        marker_sel=marker_sel,
        focus=focus,
    )

    clust_string_dict, re_gating_dict, general_dict = process_results(
        cell_data, overview, results_dictionaries, renorm_df_dict, res_in_gates, channels, cluster_string
    )

    for key in list(overview["key"]):
        generate_output(
            clust_string_dict[key],
            re_gating_dict,
            general_dict[key],
            key,
            overview,
            show_HEAT=show_HEAT,
            save_HEAT=save_HEAT,
            show_SCATTER=show_SCATTER,
            save_SCATTER=save_SCATTER,
            show_metrics_df=show_metrics_df,
            save_metrics_df=save_metrics_df,
            save_metrics_plot=save_metrics_plot,
            show_metrics_plot=show_metrics_plot,
            save_path=save_path,
        )

    return clust_string_dict, re_gating_dict, general_dict


def do_complete_gating(
    cell_data,
    channels,
    cluster_numbers,
    first_hierarchy=1,
    nr_max_hierarchies=5,
    cluster_string="louvain",
    PC=True,
    visualize_gate=True,
    learning_rate=0.05,
    iterations=50,
    nr_hyperplanes=8,
    batch_size=5000,
    grid_divisor=4,
    arange_init=[0, 2, 4, 8, 10],
    refinement_grid_search=4,
    weight_version=1,
    show_everything=False,
    max_try=5,
    marker_sel="heuristic",
    focus = "f1",
):
    """
    Parameters
    ----------
    cell_data : pd.DataFrame
        size (n_cells,n_markers) -> output of preprocess_adata_gating
    channels : list
        list of strings to indicate which channels to use for gating
    cluster_numbers : list of strings
        list of clusters to find gating strategy (e.g. ['2','4'])
    first_hierarchy : int
        start hierarch. The default is 1.
    nr_max_hierarchies : int
        maximal hierarchies in gating strategy. The default is 5.
    cluster_string : string
        name of column in cell_data that contains labels. The default is 'louvain'.
    PC : TYPE, optional
        DESCRIPTION. The default is True.

    """
    # cell_data: dataFrame with marker values per cell -> output of adata_to_df_gating
    # channels: list of marker names to consider e.g. ['marker1','marker2']
    # cluster_numbers: list of cluster_names to find gating strategy for, e.g ['1','3']
    # first_hierarchy: int -> default 1 -> start hierarchy number
    # nr_max_hierarchies: int -> default 5 -> maximum number of hierarchies in gating strategy
    # cluster_string: string, (column)name of cluster labels -> default 'louvain'
    # PC: True or False: True -> use PCA for finding optimal gate (default
    #                   False -> no PCA
    # visualize_gate: True or False -> gate visualization
    # learning_rate: double, learning rate in SGD -> default 0.05
    # iterations: int, number of iterations in SGD -> default 50
    # nr_hyperplanes: int, number of hyperplanes to specify gate -> default 8
    # batch_size: int, batch_size -> default 5000
    # grid_divisor: int, parameter for adaptive grid search -> default 4
    # arange_init: initial grid for adaptive grid search -> default [0,2,4,8,10]
    # refinement_grid_search: parameter for adaptive grid search -> default 4
    # show_everything: True if gating procedure should be visualized, False if not
    # weight_version: 0,1,2 -> if 0 : negative proportional
    #                      -> if 1 : negative proportional only if more non-targets than targets -> otherwise equal weight (default)
    #                      -> if 2 : always equal weight
    # max_try: int, number of maximal try's if internal errors occured
    # marker_sel: string, marker selection method either 'heuristic' (default)
    #                                                   'tree' (based on Decision Tree)
    #                                                   'svm'  (based on Linear SVM)
    #focus: "f1" or "recall" -> default focus "f1", optional "recall"
    # outputs:
    # result_grid_df: data frame with result and information on hyperparameters
    # results_dictionaries: dictionary -> precise information on gate locations per hierarchy

    # cluster_string = 'louvain'
    result_grid_df = pd.DataFrame(
        columns=[
            "key",
            "nr_max_hierarchies",
            "learning_rate",
            "iterations",
            "nr_hyperplanes",
            "batch_size",
            "grid_divisor",
            "arange_init",
            "refinement_grid_search",
            "weight_version",
            "cluster_number",
            "f1",
            "best_hierarchy",
        ]
    )
    renorm_df_dict = {}
    results_dictionaries = {}
    res_in_gates = {}
    #print(focus)
    for key, cluster_number in enumerate(cluster_numbers):
        # print(key)
        fail = 0
        count = 0
        while (fail == 0) & (count < max_try):
            # print('Checkpoint 2')
            if focus == "f1":
                try:
                    (
                        f1,
                        best_hierarchy,
                        summary_gating_dicts,
                        summary_losses_dicts,
                        target_df_dicts,
                        renorm_df,
                        res_in_gate,
                    ) = do_adaptive_grid_search(
                        cell_data,
                        channels,
                        cluster_number,
                        nr_max_hierarchies=nr_max_hierarchies,
                        cluster_string=cluster_string,
                        visualize_gate=visualize_gate,
                        PC=PC,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        nr_hyperplanes=nr_hyperplanes,
                        batch_size=batch_size,
                        grid_divisor=grid_divisor,
                        arange_init=arange_init,
                        refinement_grid_search=refinement_grid_search,
                        weight_version=weight_version,
                        marker_sel=marker_sel,
                        show_everything=show_everything,
                    )
                    # write results to dictionary
                    res_tmp = {
                        "key": key,
                        "nr_max_hierarchies": nr_max_hierarchies,
                        "learning_rate": learning_rate,
                        "iterations": iterations,
                        "nr_hyperplanes": nr_hyperplanes,
                        "batch_size": batch_size,
                        "grid_divisor": grid_divisor,
                        "arange_init": arange_init,
                        "refinement_grid_search": refinement_grid_search,
                        "weight_version": weight_version,
                        "cluster_number": cluster_number,
                        "f1": f1,
                        "best_hierarchy": best_hierarchy,
                    }
        
                    result_grid_df = result_grid_df.append(res_tmp, ignore_index=True)
        
                    results_dictionaries[key] = [target_df_dicts, summary_gating_dicts]
        
                    renorm_df_dict[key] = renorm_df.copy()
                    res_in_gates[key] = res_in_gate.copy()
                    # key += 1
                    # quit while-loop
                    fail = 1
                    # print(f1)
                except Exception:
                    count += 1
            if count == max_try:
                print("failed")                
                    
            if focus == "recall":
                try:
                    (
                        f1,
                        best_hierarchy,
                        summary_gating_dicts,
                        summary_losses_dicts,
                        target_df_dicts,
                        renorm_df,
                        res_in_gate,
                    ) = do_adaptive_grid_search_recall_focus(
                        cell_data,
                        channels,
                        cluster_number,
                        nr_max_hierarchies=nr_max_hierarchies,
                        cluster_string=cluster_string,
                        visualize_gate=visualize_gate,
                        PC=PC,
                        learning_rate=learning_rate,
                        iterations=iterations,
                        nr_hyperplanes=nr_hyperplanes,
                        batch_size=batch_size,
                        grid_divisor=grid_divisor,
                        arange_init=arange_init,
                        refinement_grid_search=refinement_grid_search,
                        weight_version=weight_version,
                        marker_sel=marker_sel,
                        show_everything=show_everything,
                    )
                

                    # write results to dictionary
                    res_tmp = {
                        "key": key,
                        "nr_max_hierarchies": nr_max_hierarchies,
                        "learning_rate": learning_rate,
                        "iterations": iterations,
                        "nr_hyperplanes": nr_hyperplanes,
                        "batch_size": batch_size,
                        "grid_divisor": grid_divisor,
                        "arange_init": arange_init,
                        "refinement_grid_search": refinement_grid_search,
                        "weight_version": weight_version,
                        "cluster_number": cluster_number,
                        "f1": f1,
                        "best_hierarchy": best_hierarchy,
                    }
    
                    result_grid_df = result_grid_df.append(res_tmp, ignore_index=True)
    
                    results_dictionaries[key] = [target_df_dicts, summary_gating_dicts]
    
                    renorm_df_dict[key] = renorm_df.copy()
                    res_in_gates[key] = res_in_gate.copy()
                    # key += 1
                    # quit while-loop
                    fail = 1
                    # print(f1)
                except Exception:
                    count += 1
                
           
        # print('Checkpoint 3')

    return result_grid_df, results_dictionaries, renorm_df_dict, res_in_gates


def do_adaptive_grid_search(  # noqa: max-complexity: 19
    cell_data,
    channels,
    cluster_number,
    first_hierarchy=1,
    nr_max_hierarchies=5,
    cluster_string="louvain",
    PC=True,
    visualize_gate=False,
    learning_rate=0.05,
    iterations=50,
    nr_hyperplanes=8,
    batch_size=5000,
    grid_divisor=2,
    arange_init=[0, 2, 4, 8, 10],
    refinement_grid_search=2,
    weight_version=1,
    marker_sel="heuristic",
    show_everything=True,
):
    """
    Recipe function for adaptive grid search to find an optimal gate

    Parameters
    ----------
    cell_data : pd.DataFrame
        with columns markers and cluster_string, rows -> cell values.
    channels : list
        a list of strings to indicate which channels to use for gating.
    cluster_number : str
        categories of cluster_string.
    first_hierarchy : TYPE, optional
        DESCRIPTION. The default is 1.
    nr_max_hierarchies : TYPE, optional
        DESCRIPTION. The default is 5.
    cluster_string : TYPE, optional
        DESCRIPTION. The default is 'louvain'.
    PC : TYPE, optional
        DESCRIPTION. The default is True.
    visualize_gate : TYPE, optional
        DESCRIPTION. The default is False.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.05.
    iterations : TYPE, optional
        DESCRIPTION. The default is 50.
    nr_hyperplanes : TYPE, optional
        DESCRIPTION. The default is 8.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 5000.
    grid_divisor : TYPE, optional
        DESCRIPTION. The default is 2.
    arange_init : TYPE, optional
        DESCRIPTION. The default is [0,2,4,8,10].
    refinement_grid_search : TYPE, optional
        DESCRIPTION. The default is 2.
    weight_version : int, optional
        0,1,2 -> if 0 : negative proportional
              -> if 1 : negative proportional only if more non-targets than targets -> otherwise equal weight
              -> if 2 : always equal weight. The default is 1.
    marker_sel : TYPE, optional
        DESCRIPTION. The default is 'heuristic'.
    show_everything : boolean, optional
        True if gating procedure should be visualized, False if not. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.
    best_hierarchy : list
        list of hierarchies for an optimal gating panel.
    summary_gating_dicts : dict
        summary of gating results per hierarchy.
    summary_losses_dicts : dict
        summary of the loss function results per hierarchy.
    target_df_dicts : dict of pandas DataFrames
        indicating whether a cell is part of a gate or not.
    renorm_df : TYPE
        DESCRIPTION.
    in_gate_dicts : TYPE
        DESCRIPTION.

    """

    filtered_obs = cell_data.copy()
    renorm_df = normalization(filtered_obs, channels)
    filtered_obs["label"] = (filtered_obs[cluster_string] == cluster_number) * 1
    nr_total_targets_beginning = sum(filtered_obs["label"] == 1) * 1
    combos_so_far = []
    summary_gating_dicts = {}
    in_gate_dicts = {}
    summary_losses_dicts = {}
    target_df_dicts = {}

    # initialise
    gating_dict = {}
    losses_dict = {}
    in_gate = (filtered_obs[cluster_string] == cluster_number) * 0

    for hierarchy_nr in list(np.arange(first_hierarchy, nr_max_hierarchies + 1, 1)):
        # print('Search markers')
        try:
            if marker_sel == "heuristic":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo(filtered_obs, channels)
                # end_time = time.time()
                # print('heuristic method : ' + str(np.round(end_time-start_time,4)) + ' sec')
            if marker_sel == "tree":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo_single_tree(filtered_obs, channels)
                # end_time = time.time()
                # print('tree method : ' + str(np.round(end_time-start_time,4)) + ' sec')
            if marker_sel == "svm":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo_single_svm(filtered_obs, channels)
                # end_time = time.time()
                # print('svm method : ' + str(np.round(end_time-start_time,4)) + ' sec')
        except Exception:
            hierarchy_nr -= 1
            break
        new_markers = get_new_marker_combo(best_marker_combos, combos_so_far)
        # print('Found new markers')
        marker1, marker2 = new_markers[0], new_markers[1]
        combos_so_far.append(new_markers)
        filtered_obs_temporary = filtered_obs[[marker1, marker2, "label", "cell_ID"]]
        filtered_obs_temporary.rename(columns={"label": cluster_string}, inplace=True)
        weight = fraction_targets_vanilla(filtered_obs_temporary, cluster_string)
        if weight_version == 1:
            if weight < 0:
                weight = 0
        if weight_version == 2:
            weight = 0

        current_best_f1 = -1
        try:
            arange = arange_init.copy()
            for _ in range(refinement_grid_search):
                # print('------------------------------------')
                # print('Start new hierarchy')
                # print('------------------------------------')
                recall_list = []
                f1_list = []
                precision_list = []
                # add_on = 0
                for scale in arange:
                    try:
                        in_gate_cand, losses_dict_cand, gating_dict_cand = find_2D_gate(
                            adata=filtered_obs_temporary,
                            marker1=marker1,
                            marker2=marker2,
                            cluster_number=1,
                            cluster_method=cluster_string,
                            iterations=iterations,
                            learning_rate=learning_rate,
                            nr_hyperplanes=nr_hyperplanes,
                            weight_factor_target=weight,
                            batch_size=batch_size,
                            visualize=show_everything,
                            penalty_parameter2=scale,
                            penalty_parameter=scale,
                            PC=PC,
                        )
                        # total_targets = sum(filtered_obs_temporary['louvain'] == 1)
                        recall = gating_dict_cand["tn_fp_fn_tp"][0][3] / nr_total_targets_beginning
                        precision = gating_dict_cand["tn_fp_fn_tp"][0][3] / (
                            gating_dict_cand["tn_fp_fn_tp"][0][3] + gating_dict_cand["tn_fp_fn_tp"][0][1]
                        )
                        f1 = 2 * (recall * precision) / (recall + precision)
                        recall_list.append(recall)
                        precision_list.append(precision)
                        f1_list.append(f1)
                        if f1 > current_best_f1:
                            in_gate = in_gate_cand.copy()
                            losses_dict = losses_dict_cand.copy()
                            gating_dict = gating_dict_cand.copy()
                            current_best_f1 = f1.copy()
                    except Exception:
                        pass

                cleaned_list = list(np.array(f1_list)[np.logical_not(np.isnan(np.array(f1_list)))])
                max_val = np.sort(cleaned_list)[-1]
                second_max_val = np.sort(cleaned_list)[-2]
                index2 = f1_list.index(second_max_val)
                index1 = f1_list.index(max_val)
                bound1 = arange[index1]
                bound2 = arange[index2]
                if bound1 > bound2:
                    arange = np.arange(
                        bound2 - (bound1 - bound2) / grid_divisor,
                        bound1 + 2 * (bound1 - bound2) / grid_divisor,
                        (bound1 - bound2) / grid_divisor,
                    )
                else:
                    arange = np.arange(
                        bound1 - (bound1 - bound2) / grid_divisor,
                        bound2 + 2 * (bound2 - bound1) / grid_divisor,
                        (bound2 - bound1) / grid_divisor,
                    )
                #best_penalty_strength = bound1
                #print(best_penalty_strength)
        except Exception:
            pass

        summary_gating_dicts[str(hierarchy_nr)] = gating_dict
        summary_losses_dicts[str(hierarchy_nr)] = losses_dict
        in_gate_dicts[str(hierarchy_nr)] = in_gate.copy()
        target_df_dicts[str(hierarchy_nr)] = create_target_df(
            filtered_obs, marker1, marker2, cluster_string, cluster_number
        )
        filtered_obs["gate"] = in_gate * 1
        filtered_obs = filtered_obs[filtered_obs["gate"] == 1]
        # print('hierarchy ' +str(hierarchy_nr) +' finished')
    f1_total_list = only_f1(target_df_dicts, summary_gating_dicts, hierarchy_nr)
    # best_f1_per_cluster.append(np.max(np.array(f1_total_list)))
    best_hierarchy = f1_total_list.index(max(f1_total_list)) + 1
    return (
        max(f1_total_list),
        best_hierarchy,
        summary_gating_dicts,
        summary_losses_dicts,
        target_df_dicts,
        renorm_df,
        in_gate_dicts,
    )


def find_2D_gate(
    adata,
    marker1,
    marker2,
    cluster_number,
    cluster_method="louvain",
    learning_rate=0.1,
    nr_hyperplanes=5,
    batch_size=1000,
    weight_factor_target=4,
    iterations=200,
    visualize=True,
    penalty_parameter=6,
    penalty_parameter2=0.05,
    s_sigmoid=40,
    PC=True,
):
    """
    function to get optimal location of gate for a given marker combination

    Parameters
    ----------
    adata : pd.DataFrame
        marker values and cluster assignment as columns
    marker1 : string
        name of marker1 (column of adata)
    marker2 : string
        name of marker2 (column of adata)
    cluster_number : string/int/double
        identifier of target cluster
    cluster_method : string
       name of column with cluster assignments e.g 'louvain'
    nr_hyperplanes : int
        number of hyperplanes to use for gate
    weight_factor_target : double or int
       targets contribute to loss with factor 2^(weight_factor_target) compared to non-targets
   iterations : int
        number of iterations in GradientDescentMulti
    penalty_parameter : double
        hyperparameter for regularization loss (penalty for plane away from targets)
    penalty_parameter2 : double
        hyperparamter for penalty term for collinearity of hyperplanes
    s_sigmoid : int or double
        scaling factor in sigmoid function

    Returns
    -------
    in_gate : np.array - shape (nr_of_cells,)
        one-hot vector indicating which cells fall in gate
    final_dict : dict
        summary of GradientDescentMulti
    gating_dict : dict
        summary of gates and further information




    :param iterations: int number of iterations in GradientDescentMulti
    :param visualize: boolean, True => plot of gate and cells, False => no plot of gate and cells
    :param penalty_parameter: double, hyperparameter for regularization loss (penalty for plane away from targets)
    :param penalty_parameter2: double, hyperparamter for penalty term for collinearity of hyperplanes
    :param s_sigmoid: int or double, scaling factor in sigmoid function
    :return:
        :in_gate: np.array of shape (nr_of_cells,) - shows which cells fall into gate
        :final_dict: summary dictionary of GradientDescentMulti
        :gating_dict: summary of gates and further information
    """
    # prepare gradient descent
    target_df = create_target_df(adata, marker1, marker2, cluster_method, cluster_number)
    markers = target_df.values[:, 0:2].T
    label = target_df.values[:, 2]
    weighting = (label + 1) ** weight_factor_target
    marker_for_pca = target_df[target_df["label"] == 1].values[:, 0:2]
    normal_vectors, biases = initialize_norm_bias(nr_hyperplanes, marker_for_pca, PC=PC)
    # filtered_target = target_df[target_df["label"] == 1]

    # do gradient descent
    final_dict = GradientDescentMulti(
        target_df,
        normal_vectors,
        biases,
        learning_rate,
        iterations,
        markers,
        label,
        weighting,
        s=s_sigmoid,
        batch_size=batch_size,
        penalty_parameter=penalty_parameter,
        penalty_parameter2=penalty_parameter2,
    )
    # proecessing of results
    norm_vec_list, biases_list = get_normal_v_biases(final_dict, nr_hyperplanes)
    norm_unit_square, bias_unit_square = generate2unit_planes()
    norm_vec_list.extend(norm_unit_square)
    biases_list.extend(bias_unit_square)

    # calculate corners of polygon based on normal vectors
    out = get_two_points_each_hyperplane(norm_vec_list, biases_list)
    candidates = get_candidate_points(out)
    fin_points = get_relevant_points(norm_vec_list, biases_list, candidates)
    points_connect = get_points_to_connect(norm_vec_list, biases_list, fin_points)

    # classify points (hard margin)
    in_gate = classify_points(norm_vec_list, biases_list, markers)

    # create summary gating dictionary
    gating_dict = {}
    gating_dict["marker_combo"] = [marker1, marker2]
    gating_dict["gate_points"] = fin_points
    gating_dict["gate_edges"] = points_connect
    gating_dict["normal_vectors"] = norm_vec_list
    gating_dict["biases"] = biases_list
    # append true negatives, false positives, false negatives and true positives to gating dictionary
    gating_dict["tn_fp_fn_tp"] = [sklearn.metrics.confusion_matrix(target_df["label"], in_gate).ravel()]
    return in_gate, final_dict, gating_dict


def generate_output(
    clust_string,
    re_gating_dict,
    general,
    key,
    overview,
    show_HEAT=True,
    save_HEAT=True,
    show_SCATTER=True,
    save_SCATTER=True,
    show_metrics_df=True,
    save_metrics_df=True,
    show_metrics_plot=True,
    save_metrics_plot=True,
    save_path=os.getcwd(),
):
    n_h, hull_dict, path_dict, current_gate_dicts, gate_points_dict = apply_convex_hull(general, key, re_gating_dict)
    general_targ = general[general["true_label"] == 1]
    general_non_targ = general[general["true_label"] == 0]
    scores = metrics_new(
        clust_string,
        general,
        n_h,
        show_metrics_df=show_metrics_df,
        save_metrics_df=save_metrics_df,
        save_path=save_path,
    )
    best_hierarchy = np.argmax(scores.T["f1"]) + 1
    do_plot_metrics(
        clust_string,
        scores,
        key,
        overview,
        save_metrics_plot=save_metrics_plot,
        show_metrics_plot=show_metrics_plot,
        save_path=save_path,
    )

    for hierarch in range(1, best_hierarchy + 1):
        xlim, ylim = do_SCATTER(
            clust_string,
            general,
            general_targ,
            general_non_targ,
            hierarch,
            re_gating_dict,
            hull_dict,
            gate_points_dict,
            key,
            show_SCATTER=show_SCATTER,
            save_SCATTER=save_SCATTER,
            save_path=save_path,
        )

        do_HEAT_targets(
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
            show_HEAT=show_HEAT,
            save_HEAT=save_HEAT,
            save_path=save_path,
        )

        do_HEAT_non_targets(
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
            show_HEAT=show_HEAT,
            save_HEAT=save_HEAT,
            save_path=save_path,
        )

def convex_hull_add_on(meta_info_path,target_location,add_summary = True):
    #meta_info_path: path to folder where meta info is saved
    #target_location: path to folder where results with convex hull should be saved
    meta_info = np.load(meta_info_path,allow_pickle=True).item()
    cluster_IDs = list(meta_info['clusterkeys'].keys())
    cluster_names = list((meta_info['clusterkeys'].values()))
    base_df_dict = {}
    for cluster_ID in cluster_IDs:
        base_df = add_tight_analysis(meta_info,cluster_ID,os.path.join(target_location, 'cluster_' + cluster_names[cluster_ID]))
        base_df_dict[meta_info['clusterkeys'][cluster_ID]] = base_df.filter(regex=r'^gate_hull_(?!0)')
        plot_metric_tight(meta_info,cluster_ID,os.path.join(target_location, 'cluster_' + cluster_names[cluster_ID]),save=True,show=True)
    np.save(os.path.join(target_location, "info_gate_membership.npy"), base_df_dict)
    if add_summary:
        make_performance_summary(meta_info_path = meta_info_path,target_location=target_location)
        make_marker_summary(meta_info_path,target_location=target_location)

def do_adaptive_grid_search_recall_focus(  # noqa: max-complexity: 19
    cell_data,
    channels,
    cluster_number,
    first_hierarchy=1,
    nr_max_hierarchies=5,
    cluster_string="louvain",
    PC=True,
    visualize_gate=False,
    learning_rate=0.05,
    iterations=50,
    nr_hyperplanes=8,
    batch_size=5000,
    grid_divisor=2,
    arange_init=[0, 2, 4, 8, 10],
    refinement_grid_search=2,
    weight_version=1,
    marker_sel="heuristic",
    show_everything=True,
):
    """
    Recipe function for adaptive grid search to find an optimal gate

    Parameters
    ----------
    cell_data : pd.DataFrame
        with columns markers and cluster_string, rows -> cell values.
    channels : list
        a list of strings to indicate which channels to use for gating.
    cluster_number : str
        categories of cluster_string.
    first_hierarchy : TYPE, optional
        DESCRIPTION. The default is 1.
    nr_max_hierarchies : TYPE, optional
        DESCRIPTION. The default is 5.
    cluster_string : TYPE, optional
        DESCRIPTION. The default is 'louvain'.
    PC : TYPE, optional
        DESCRIPTION. The default is True.
    visualize_gate : TYPE, optional
        DESCRIPTION. The default is False.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.05.
    iterations : TYPE, optional
        DESCRIPTION. The default is 50.
    nr_hyperplanes : TYPE, optional
        DESCRIPTION. The default is 8.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 5000.
    grid_divisor : TYPE, optional
        DESCRIPTION. The default is 2.
    arange_init : TYPE, optional
        DESCRIPTION. The default is [0,2,4,8,10].
    refinement_grid_search : TYPE, optional
        DESCRIPTION. The default is 2.
    weight_version : int, optional
        0,1,2 -> if 0 : negative proportional
              -> if 1 : negative proportional only if more non-targets than targets -> otherwise equal weight
              -> if 2 : always equal weight. The default is 1.
    marker_sel : TYPE, optional
        DESCRIPTION. The default is 'heuristic'.
    show_everything : boolean, optional
        True if gating procedure should be visualized, False if not. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.
    best_hierarchy : list
        list of hierarchies for an optimal gating panel.
    summary_gating_dicts : dict
        summary of gating results per hierarchy.
    summary_losses_dicts : dict
        summary of the loss function results per hierarchy.
    target_df_dicts : dict of pandas DataFrames
        indicating whether a cell is part of a gate or not.
    renorm_df : TYPE
        DESCRIPTION.
    in_gate_dicts : TYPE
        DESCRIPTION.

    """

    filtered_obs = cell_data.copy()
    renorm_df = normalization(filtered_obs, channels)
    filtered_obs["label"] = (filtered_obs[cluster_string] == cluster_number) * 1
    nr_total_targets_beginning = sum(filtered_obs["label"] == 1) * 1
    combos_so_far = []
    summary_gating_dicts = {}
    in_gate_dicts = {}
    summary_losses_dicts = {}
    target_df_dicts = {}

    # initialise
    gating_dict = {}
    losses_dict = {}
    in_gate = (filtered_obs[cluster_string] == cluster_number) * 0

    for hierarchy_nr in list(np.arange(first_hierarchy, nr_max_hierarchies + 1, 1)):
        # print('Search markers')
        try:
            if marker_sel == "heuristic":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo(filtered_obs, channels)
                # end_time = time.time()
                # print('heuristic method : ' + str(np.round(end_time-start_time,4)) + ' sec')
            if marker_sel == "tree":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo_single_tree(filtered_obs, channels)
                # end_time = time.time()
                # print('tree method : ' + str(np.round(end_time-start_time,4)) + ' sec')
            if marker_sel == "svm":
                # start_time = time.time()
                best_marker_combos = return_best_marker_combo_single_svm(filtered_obs, channels)
                # end_time = time.time()
                # print('svm method : ' + str(np.round(end_time-start_time,4)) + ' sec')
        except Exception:
            hierarchy_nr -= 1
            break
        new_markers = get_new_marker_combo(best_marker_combos, combos_so_far)
        # print('Found new markers')
        marker1, marker2 = new_markers[0], new_markers[1]
        combos_so_far.append(new_markers)
        filtered_obs_temporary = filtered_obs[[marker1, marker2, "label", "cell_ID"]]
        filtered_obs_temporary.rename(columns={"label": cluster_string}, inplace=True)
        weight = fraction_targets_vanilla(filtered_obs_temporary, cluster_string)
        if weight_version == 1:
            if weight < 0:
                weight = 0
        if weight_version == 2:
            weight = 0

        current_best_f1 = -1
        current_best_recall = -1
        try:
            arange = arange_init.copy()
            for _ in range(refinement_grid_search):
                # print('------------------------------------')
                # print('Start new hierarchy')
                # print('------------------------------------')
                recall_list = []
                f1_list = []
                precision_list = []
                # add_on = 0
                for scale in arange:
                    try:
                        in_gate_cand, losses_dict_cand, gating_dict_cand = find_2D_gate(
                            adata=filtered_obs_temporary,
                            marker1=marker1,
                            marker2=marker2,
                            cluster_number=1,
                            cluster_method=cluster_string,
                            iterations=iterations,
                            learning_rate=learning_rate,
                            nr_hyperplanes=nr_hyperplanes,
                            weight_factor_target=weight,
                            batch_size=batch_size,
                            visualize=show_everything,
                            penalty_parameter2=scale,
                            penalty_parameter=scale,
                            PC=PC,
                        )
                        # total_targets = sum(filtered_obs_temporary['louvain'] == 1)
                        recall = gating_dict_cand["tn_fp_fn_tp"][0][3] / nr_total_targets_beginning
                        precision = gating_dict_cand["tn_fp_fn_tp"][0][3] / (
                            gating_dict_cand["tn_fp_fn_tp"][0][3] + gating_dict_cand["tn_fp_fn_tp"][0][1]
                        )
                        f1 = 2 * (recall * precision) / (recall + precision)
                        recall_list.append(recall)
                        precision_list.append(precision)
                        f1_list.append(f1)
                        if recall > current_best_recall:
                            in_gate = in_gate_cand.copy()
                            losses_dict = losses_dict_cand.copy()
                            gating_dict = gating_dict_cand.copy()
                            current_best_recall = recall.copy()
                    except Exception:
                        pass

                cleaned_list = list(np.array(recall_list)[np.logical_not(np.isnan(np.array(recall_list)))])
                max_val = np.sort(cleaned_list)[-1]
                second_max_val = np.sort(cleaned_list)[-2]
                index2 = recall_list.index(second_max_val)
                index1 = recall_list.index(max_val)
                bound1 = arange[index1]
                bound2 = arange[index2]
                if bound1 > bound2:
                    arange = np.arange(
                        bound2 - (bound1 - bound2) / grid_divisor,
                        bound1 + 2 * (bound1 - bound2) / grid_divisor,
                        (bound1 - bound2) / grid_divisor,
                    )
                else:
                    arange = np.arange(
                        bound1 - (bound1 - bound2) / grid_divisor,
                        bound2 + 2 * (bound2 - bound1) / grid_divisor,
                        (bound2 - bound1) / grid_divisor,
                    )
                #best_penalty_strength = bound1
                #print(best_penalty_strength)
        except Exception:
            pass

        summary_gating_dicts[str(hierarchy_nr)] = gating_dict
        summary_losses_dicts[str(hierarchy_nr)] = losses_dict
        in_gate_dicts[str(hierarchy_nr)] = in_gate.copy()
        target_df_dicts[str(hierarchy_nr)] = create_target_df(
            filtered_obs, marker1, marker2, cluster_string, cluster_number
        )
        filtered_obs["gate"] = in_gate * 1
        filtered_obs = filtered_obs[filtered_obs["gate"] == 1]
        # print('hierarchy ' +str(hierarchy_nr) +' finished')
    f1_total_list = only_f1(target_df_dicts, summary_gating_dicts, hierarchy_nr)
    # best_f1_per_cluster.append(np.max(np.array(f1_total_list)))
    best_hierarchy = f1_total_list.index(max(f1_total_list)) + 1
    return (
        max(f1_total_list),
        best_hierarchy,
        summary_gating_dicts,
        summary_losses_dicts,
        target_df_dicts,
        renorm_df,
        in_gate_dicts,
    )
    

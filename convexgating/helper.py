#!/usr/bin/env python
import os
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import scanpy as sc
import anndata as ann
import os
import re
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def adata_to_df_gating(adata, cluster_string):
    """
    transform adata such that in right form for gating algorithm

    Parameters
    ----------
    adata : anndata object
        annotated data frame as data input.
    cluster_string : str
        needs to be a column name of the adata.obs metadata.

    Returns
    -------
    cell_data : pandas DataFrame
        prepared DataFrame as input for the gating algorithm.

    """

    channels = list(adata.var.index)
    if type(adata.X) != np.ndarray:
        cell_data = pd.DataFrame(adata.X.todense(), columns=channels)
        cell_data[cluster_string] = adata.obs[cluster_string].values
    else:
        cell_data = pd.DataFrame(adata.X, columns=channels)
        cell_data[cluster_string] = adata.obs[cluster_string].values
    return cell_data


def return_best_marker_combo(filtered_obs, channels):
    """
    Function to compute the best marker combination
    to gate upon using a distribution heuristic
    :param filtered_obs: dataframe with marker values and
                         column 'label' 0-1 encoded whether
                         cell is target or non-target cell
    :param channels: list of columns in 'filtered_obs' that are relevant markers
    :return:
        a dataframe of 'channels' and the corresponding result of the heuristic
    """

    targetcells_df = filtered_obs[filtered_obs["label"] == 1]
    non_targetcells_df = filtered_obs[filtered_obs["label"] == 0]
    target_greedy_df, nontarget_greedy_df = marker_greedy_summary(targetcells_df, non_targetcells_df, channels)
    marker_heuristic = heuristic_markers(target_greedy_df, nontarget_greedy_df, channels)
    return marker_heuristic


def normalization(adata, channels):
    """
    Normalizes columns of a data frame, corresponding value will lie between 0 and 1
    :param adata: data frame with marker values and cluster assignment as columns
    :param channels: list of strings of markers, e.g channels = ['FSC-H','FSC-A','SSC-H']
    :return:
        data frame with min and max value per marker -> will be needed to recover the original data
    """
    # minima = []
    # maxima = []
    df_min_max = pd.DataFrame(index=["min", "max"])
    for string in channels:
        norm, minimum, maximum = normalize_column(adata[string].values)
        adata[string] = norm
        df_min_max[string] = [minimum, maximum]
    return df_min_max


def possible_marker_combinations(channels):
    """
    generates dictionary with all possible marker combinations
    :param channels: list of strings containing markers
    :return:
        channels_combination_dict
    """
    nr_channels = len(channels)
    channels_combination_dict = {}
    counter = 0
    for j in range(nr_channels):
        for s in range(j + 1, nr_channels):
            channels_combination_dict[counter] = [channels[j], channels[s]]
            counter += 1
    return channels_combination_dict


def normalize_column(values):
    """
    Normalizes an array s.t. values lie between 0 and 1
    :param values: np.array of shape (n,)
    :return:
        :param norm: normalized np.array of shape (n,)
        :param minimum: minimal value
        :param maximum: maximal value
    """
    minimum = min(values) - 0.001
    maximum = max(values) + 0.001
    norm = (values - minimum) / (maximum - minimum)
    return norm, minimum, maximum


def renormalize_column(values, minimum, maximum):
    """
    reverts the normalization of the function 'normalize_column'
    :param values: np.array of shape (n,)
    :param minimum: min value in original data
    :param maximum: max value in original data
    :return:
        :param orig_scale: np.array of shape (n,) renormalized array
    """
    orig_scale = values * (maximum - minimum) + minimum
    return orig_scale


def summary_metrics(out_df, nr_total_hierarch):
    """
    Summary of the gating strategy prediction.

    Parameters
    ----------
    out_df : pandas DataFrame
        Output data frame of the inferred gating strategy.
    nr_total_hierarch : int
        Number of hierarchies used in the gating strategy.

    Returns
    -------
    f1_rec_prec_out : pandas DataFrame
        Contains the f1 score, precision and recall as summary metrics

    """
    f1_rec_prec_out = pd.DataFrame(index=["f1", "recall", "precision"])
    for hier in range(1, nr_total_hierarch + 1):
        g_string = "gate_" + str(hier)
        h_string = "hierarchy_" + str(hier)
        f = np.round(f1_score(out_df["true_label"].values, out_df[g_string].values), 6)
        r = np.round(recall_score(out_df["true_label"].values, out_df[g_string].values), 6)
        p = np.round(precision_score(out_df["true_label"].values, out_df[g_string].values), 6)
        f1_rec_prec_out[h_string] = f, r, p
    return f1_rec_prec_out


def create_target_df(adata, marker_string1, marker_string2, cluster_string, target_cluster):
    """
    creates a dataFrame with only two selected markers and the corresponding label
    :param adata: data frame with marker values and cluster assignment as columns
    :param marker_string1: string, name of marker1 (column of adata)
    :param marker_string2: string, name of marker2 (column of adata)
    :param cluster_string: string, name of column with cluster assignments e.g 'louvain'
    :param target_cluster: string or int or double, identifier of target cluster (depends on values )
    :return:
        pandas data frame with two markers and one-hot encoded column label that tells whether
        specific cell is a target or non-target cell
    """
    marker1 = adata[marker_string1].values
    marker2 = adata[marker_string2].values
    targets_one_hot = (adata[cluster_string].values == target_cluster) * 1
    cell_ID = adata["cell_ID"].values

    output_df = pd.DataFrame(
        {marker_string1: marker1, marker_string2: marker2, "label": targets_one_hot, "cell_ID": cell_ID}
    )

    return output_df


def target_summary(labels, target_label):
    """
    Computes the number of cells with the target label and the
    total number of cells
    :param labels: shape (n_cells,)
    :return:
        number of target cells, number of total cells
    """
    return int(sum((labels == target_label) * 1.0)), len(labels)


def line_intersect(a1, a2, b1, b2):
    """
    returns intersection point of two lines in 2D that
    are defined via two points each
    :param a1: np.array of shape (2,), point on first line
    :param a2: np.array of shape (2,), point on first line
    :param b1 = np.array of shape (2,), point on second line
    :param b1 = np.array of shape (2,), point on second line
    :return:
        np.array of shape (2,), intersection point of the two lines
    """
    T = np.array([[0, -1], [1, 0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


def two_points_hyperplane2D(normal_vector, bias):
    """
    calculates two points on a hyplerplane in in 2D that is specified via a normal vector and
    a bias, that is normal_vector@x + bias = 0
    :param normal_vector: np.array of shape (2,) but not np.array([0,0])
    :param bias: scalar (double)
    :return:
       tuple (two components np.arrays of shape (2,)), two points on the hyperplane
    """
    assert sum(normal_vector != np.array([0, 0])) != 0

    if sum(normal_vector != np.array([0, 0])) == 2:
        A = np.array([[normal_vector[0], normal_vector[1]], [-normal_vector[0], normal_vector[1]]])
        b1 = np.array([bias, bias + 1])
        b2 = np.array([bias, bias - 1])
        p1 = np.linalg.lstsq(A, -b1)[0]
        p2 = np.linalg.lstsq(A, -b2)[0]
        return p1, p2

    if normal_vector[0] == 0:
        p1 = [0, -bias / normal_vector[1]]
        p2 = [0.5, -bias / normal_vector[1]]
        return np.array(p1), np.array(p2)

    if normal_vector[1] == 0:
        p1 = [-bias / normal_vector[0], 0]
        p2 = [-bias / normal_vector[0], 0.5]
        return np.array(p1), np.array(p2)


def value_halfspace(normal_vector, bias, cell_value):
    """
    :param normal_vector: torch.tensor of torch.Size([2])
    :param bias: torch.tensor of torch.Size([1])
    :param cell_value: torch.tensor of float of torch.Size([2, M]) where M number of cells
    :return:
       torch.Size([M]) halfspace value of the cells
    """
    return normal_vector @ cell_value + bias


def get_relevant_points(normal_vectors, biases, possible_points):
    """
    among all intersection points of the hyperplanes, return only those relevant for the gate
    :param normal_vectors: list of torch.tensor of torch.Size([2])
    :param biases: list of torch.tensor of torch.Size([1])
    :param possible_points: list of np.arrays of shape (2,)
    :return:
        :param boundary_points: list of np.arrays of shape (2,), corners of gate
    """
    # nr_boundaries = len(normal_vectors)
    # nr_possible_points = len(possible_points)

    boundary_points = []
    for j in range(len(possible_points)):
        counter = 0
        for s in range(len(normal_vectors)):
            if (abs(value_halfspace(normal_vectors[s], biases[s], possible_points[j])) < 0.0001) or (
                value_halfspace(normal_vectors[s], biases[s], possible_points[j]) > 0
            ):
                counter += 1
        # print(counter)
        if counter == len(normal_vectors):
            boundary_points.append(possible_points[j])
    return boundary_points


def get_points_to_connect(normal_vectors, biases, final_points):
    """
    returns a list of lists that shows which gate points should be
    connected (for visualization)
    :param normal_vectors: list of torch.tensor of torch.Size([2])
    :param biases: list of torch.tensor of torch.Size([1])
    :param final_points: list of np.arrays of shape (2,) corners of gate
    :return:
        :param points_to_connect: list of lists where the jth list contains
        two points that should be connected
    """
    points_to_connect = []
    for j in range(len(final_points)):
        for s in range(j + 1, len(final_points)):
            for q in range(len(normal_vectors)):
                if (abs(value_halfspace(normal_vectors[q], biases[q], final_points[j])) < 0.00001) & (
                    abs(value_halfspace(normal_vectors[q], biases[q], final_points[s])) < 0.00001
                ):
                    points_to_connect.append([final_points[j], final_points[s]])
        # value_halfspace(normal_vectors[0],biases[0],final_points[1])
    return points_to_connect


def sigmoid_s(x, s):
    """
    calculates scaled version of sigmoid
    :param x: np.array
    :param s: int or double
    :return:
         scaled version of sigmoid, same shape as x
    """
    return 1 / (1 + np.exp(-s * x))


def sigmoid_s_torch(x, s):
    """
    calculates scaled version of sigmoid (torch version)
    :param x: torch.tensor of floats
    :param s: int or double
    :return:
        scaled version of sigmoid, same shape as x (torch version)
    """
    return 1 / (1 + torch.exp(-s * x))


def get_two_points_each_hyperplane(norm_vec_list, bses):
    """
    calculate two points on each hyperplane given normal vectors and biases
    :param norm_vec_list: list of np.arrays of shape (2,)
    :param bses: list of np.ararys of shape (1,)
    :return:
        res_list: list of tuples
    """
    res_list = []
    nr_H = len(norm_vec_list)
    for j in range(nr_H):
        res_list.append(two_points_hyperplane2D(norm_vec_list[j], bses[j]))
    return res_list


def get_candidate_points(points_on_planes):
    """
    given two points on each hyperplane, calculate all intersection points of the hyperplanes
    :param points_on_planes: list of tuples (each tuple (two points) determines a hyperplane)
    :return:
        candidate_points: list of np.arrays of shape (2,) - all intersection points of any of the hyperplanes
    """
    candidate_points = []
    m = len(points_on_planes)
    for j in range(m):
        for s in range(j + 1, m):
            candidate_points.append(
                line_intersect(
                    np.squeeze(points_on_planes[j][0]),
                    np.squeeze(points_on_planes[j][1]),
                    np.squeeze(points_on_planes[s][0]),
                    np.squeeze(points_on_planes[s][1]),
                )[0]
            )

    return candidate_points


def GradientDescentMulti(  # noqa: max-complexity: 13
    target_df,
    normal_vectors,
    biases,
    learning_rate,
    iterations,
    markers,
    label,
    weighting,
    s=100,
    batch_size=1000,
    penalty_parameter=0.05,
    penalty_parameter2=0,
):
    """
    performs GradientDescent to find optimal normal vectors and biases
    :param target_df: dataFrame output of create_target_df
    :param normal vectors: list of initialized normal vectors (torch.tensors) of torch.Size([2])
    :param biases: list of initialized biases (torch.tensors) of torch.Size([2])
    :param learning_rate: double
    :param iterations: int, number of total iterations
    :param markers: np.array of shape (2,nr_of_cells) marker values (equal to target_df.values[:,0:2].T)
    :param label: np.array of shape (nr_of_cells,) label (equal to target_df.values[:,2])
    :param weighting: np.array of shape (nr_of_cells,) weight factor for contribution of each individual cell to loss
    :param s: int - scaling factor in sigmoid function
    :param batch_size: int batch size per iteration
    :param penalty_parameter: double, hyperparameter for regularization loss (penalty for plane away from targets)
    :param penalty_parameter2: double, hyperparamter for penalty term for collinearity of hyperplanes
    :return:
        info_dict: dictionary with information on losses, final normal vectors and final bias
    """
    markers_orig = markers.copy()
    label_orig = label.copy()
    weighting_orig = weighting.copy()
    normal_vectors, biases = adapt_init_norm_bias(normal_vectors, biases)
    losses = []
    for _ in range(iterations):
        if markers_orig.shape[1] > batch_size:
            indices = np.random.choice(markers_orig.shape[1], batch_size)
            markers = markers_orig[:, indices]
            label = label_orig[indices]
            weighting = weighting_orig[indices]

        nr_norm_vectors = len(normal_vectors)
        halfspace_v = []
        halfspace_sigm = []
        for j in range(nr_norm_vectors):
            h = value_halfspace(normal_vectors[j], biases[j], torch.tensor(markers).float())
            halfspace_v.append(h)
            halfspace_sigm.append(sigmoid_s_torch(h, s))
        prob_in_gate = 1
        for s in range(nr_norm_vectors):
            prob_in_gate *= halfspace_sigm[s]

        mid_point_targets = get_mean_target_value(target_df)
        penalty_distance = get_penalty_distance(normal_vectors, biases, mid_point_targets)
        penalty_distance_add = penalty_1_99_targets(target_df, normal_vectors, biases, len(normal_vectors))
        # penalty_distance += penalty_distance_add

        # normalized_norm_vectors = norm1_normal_vectors(normal_vectors)
        # penalty_projection = -sum_projections(normal_vectors)

        # print(penalty_parameter)
        # print(penalty_parameter2)
        min_poss_val = -0.001
        max_poss_val = 1.001
        prob_in_gate = (prob_in_gate - min_poss_val) / (max_poss_val - min_poss_val)
        loss = torch.nn.BCELoss(weight=torch.tensor(weighting).float())
        output = (
            loss(prob_in_gate, torch.tensor(label).float())
            + penalty_parameter * penalty_distance
            + penalty_parameter2 * penalty_distance_add
        )
        # output = loss(prob_in_gate,torch.tensor(label).float()) + penalty_parameter*penalty_distance
        losses.append(output)
        # print(output)
        output.backward()

        with torch.no_grad():
            gradient_abs_sum = 0.0
            for u in range(nr_norm_vectors - 4):
                normal_vectors[u] -= learning_rate * normal_vectors[u].grad
                gradient_abs_sum += torch.norm(normal_vectors[u].grad)
            for u in range(nr_norm_vectors):
                biases[u] -= learning_rate * biases[u].grad
                gradient_abs_sum += torch.norm(biases[u].grad)
        for q in range(nr_norm_vectors - 4):
            normal_vectors[q].grad.zero_()
        for q in range(nr_norm_vectors):
            biases[q].grad.zero_()
        if gradient_abs_sum < 1.4097e-07:
            break

    info_dict = {}
    info_dict["loss"] = losses
    for x in range(1, nr_norm_vectors + 1):
        info_dict["normal_vector" + str(x)] = normal_vectors[x - 1]
        info_dict["bias" + str(x)] = biases[x - 1]
    return info_dict


def initialize_norm_bias(nr_of_hyperplanes, marker_for_pca, PC=False):
    """
    initialize normal vectors and biases randomly (the normal vectors of
    4 out of nr_of_hyperplanes hyperplanes wont be changed during training)
    :param nr_of_hyperplanes: int
    :return:
        normal_vectors: list of len nr_of_hyperplanes where each entry is
                        torch.tensor of torch.Size([2])
        biases: list of len nr_of_hyperplanes where each entry is
                torch.tensor of torch.Size([1])
    """

    init_norm_vectors = list((2 * np.random.rand(2, nr_of_hyperplanes) - 1).T)
    init_biases = list((2 * np.random.rand(1, nr_of_hyperplanes) - 1).T)
    if not PC:
        if nr_of_hyperplanes > 3:
            init_norm_vectors[-1] = np.array([0.0, 1])
            init_norm_vectors[-2] = np.array([0.0, -1])
            init_norm_vectors[-3] = np.array([1.0, 0.0])
            init_norm_vectors[-4] = np.array([-1.0, 0])
    else:
        if nr_of_hyperplanes > 3:
            pca = PCA(n_components=2)
            pca.fit(marker_for_pca)
            comp1 = pca.components_[0, :]
            comp2 = pca.components_[1, :]
            init_norm_vectors[-1] = comp1
            init_norm_vectors[-2] = -comp1
            init_norm_vectors[-3] = comp2
            init_norm_vectors[-4] = -comp2

    normal_vectors = []
    biases = []
    for j in range(nr_of_hyperplanes):
        normal_vectors.append(torch.tensor(list(init_norm_vectors[j]), requires_grad=True).float())
        biases.append(torch.tensor(list(init_biases[j]), requires_grad=True).float())
    for u in range(4):
        # normal_vectors[-(u+1)].requires_grad = False
        normal_vectors[-(u + 1)] = torch.tensor(normal_vectors[-(u + 1)].detach().numpy(), requires_grad=False)
    return normal_vectors, biases


def adapt_init_norm_bias(normal_vectors, biases):
    for z in range(len(normal_vectors)):
        if normal_vectors[z].requires_grad:
            normal_vectors[z] = torch.tensor(normal_vectors[z], requires_grad=True)
        biases[z] = torch.tensor(biases[z], requires_grad=True)
    return normal_vectors, biases


def get_normal_v_biases(final_dict, nr_hyperplanes):
    """
    given the result of the GradientDescentMulti, output optimal normal
    vectors and biases as list of np.arrays
    :param final_dict: dictionary, output of the GradientDescentMulti function
    :param nr_hyperplanes: int
    :return:
        norm_vec_list: list of len nr_hyperplanes, each entry np.array of shape (2,)
        biases: list of len nr_hyperplanes, each entry np.array of shape (1,)
    """
    norm_vec_list = []
    biases = []
    for s in range(nr_hyperplanes):
        norm_vec_list.append(final_dict["normal_vector" + str(s + 1)].detach().numpy())
        biases.append(final_dict["bias" + str(s + 1)].detach().numpy())
    return norm_vec_list, biases


def classify_points(norm_vec_list, biases_list, markers):
    """
    determine which points fall into the gate
    :param norm_vec_list: list where each entry np.array of shape (2,)
    :param biases_list: list where each entry np.array of shape (1,)
    :param markers: np.array of shape (2,nr_of_cells) - marker values
    :return:
        one_hot_in_gate: np.array of shape (nr_of_cells,) -
                         one-hot-encoded - 1 means cell in gate
    """
    nr_norm_vec = len(norm_vec_list)
    H_values = value_halfspace(np.array(norm_vec_list), np.array(biases_list), markers)
    sign_matrix = np.sign(H_values)
    one_hot_in_gate = (np.sum(sign_matrix, axis=0) > (nr_norm_vec - 0.5)) * 1
    return one_hot_in_gate


def generate2unit_planes():
    """
    get list of normal vectors and biases for square box
    with corners (0/0),(1,0),(0,1),(1,1)
    :return:
        # list of normal vectors where each entry np.array of shape (2,)
        # list of biases where each entry np.array of shape (1,)
    """
    n1 = np.array([1, 0])
    n2 = np.array([-1, 0])
    n3 = np.array([0, 1])
    n4 = np.array([0, -1])
    b1 = np.array([0])
    b2 = np.array([1])
    b3 = np.array([0])
    b4 = np.array([1])
    return [n1, n2, n3, n4], [b1, b2, b3, b4]


def print_results_current_pos_only_f1(summary_gating_dicts, first_hierarchy_level, current_hierarchy_level):
    """
    summary of the performance of the gating procedure up to the current hierarchy
    :param summary_gating_dicts: dictionary where each element is a gating dict,
                                 e.g key '1' contains gating_dict of hierarchy 1
                                     key '2' contains gating_dict of hierarchy 1
    :param first_hierarchy_level: int , first hierarchy level in gating procedure, usually 1
    :param current_hierarchy_level: int , current hierarchy level in gating procedure, >= 1
    :return:
         f1
    """
    tp_last = summary_gating_dicts[str(current_hierarchy_level)]["tn_fp_fn_tp"][0][3]
    fp_last = summary_gating_dicts[str(current_hierarchy_level)]["tn_fp_fn_tp"][0][1]
    tp_beg = summary_gating_dicts[str(first_hierarchy_level)]["tn_fp_fn_tp"][0][3]
    fn_beg = summary_gating_dicts[str(first_hierarchy_level)]["tn_fp_fn_tp"][0][2]
    # fn_last = summary_gating_dicts[str(current_hierarchy_level)]['tn_fp_fn_tp'][0][2]

    recall = np.round(tp_last / (tp_beg + fn_beg), 7)
    precision = np.round(tp_last / (tp_last + fp_last), 7)
    f1 = np.round(2 * (precision * recall) / (precision + recall), 7)

    # f1 of the last hierarchy
    # recall_last =  np.round(tp_last/(tp_last+fn_last),7)
    # f1_last = np.round(2 * (precision * recall_last) / (precision + recall_last),7)

    return f1


def marker_greedy_summary(targetcells_df, non_targetcells_df, channels):
    """
    create a summary pandas dataframe with some basic summary for targets and non-targets
    :print targetcells_df: dataframe with all marker values filtered for target values
    :print nontargetcells_df: dataframe with all marker values filtered for target values
    :return:
        target_greedy_df: pandas data frame for target cells rows: 'median','99perc','1perc' columns: markers
        nontarget_greedy_df: pandas data frame for non-target cells rows: 'median','99perc','1perc' columns: markers
    """
    nontarget_greedy_df = pd.DataFrame(index=["median", "99perc", "1perc"])
    target_greedy_df = pd.DataFrame(index=["median", "99perc", "1perc"])
    for string in channels:
        l_n_t = []

        l_n_t.append(np.median(non_targetcells_df[string]))
        l_n_t.append(np.percentile(non_targetcells_df[string], 99))
        l_n_t.append(np.percentile(non_targetcells_df[string], 1))
        nontarget_greedy_df[string] = l_n_t

        l_t = []

        l_t.append(np.median(targetcells_df[string]))
        l_t.append(np.percentile(targetcells_df[string], 99))
        l_t.append(np.percentile(targetcells_df[string], 1))
        target_greedy_df[string] = l_t
    return target_greedy_df, nontarget_greedy_df


def heuristic_markers(targetcells_df, non_targetcells_df, channels):
    """
    return a dictionary which shows which markers have "the most different"
    distribution of target and non-target cells
    :param target_greedy_df: panda dataframe (output of marker_greedy_summary)
    :param nontarget_greedy_df: panda dataframe (output of marker_greedy_summary)
    :return:
        diff_targets_dict: sorted dictionary with (key:name_of_marker, value:difference_MSE_value)
                           first entry marker with biggest difference of targets and non-targets
    """
    target_greedy_df, nontarget_greedy_df = marker_greedy_summary(targetcells_df, non_targetcells_df, channels)
    diff_targets_dict = {}
    for string in channels:
        diff_targets_dict[string] = MSE(target_greedy_df[string].values, nontarget_greedy_df[string].values)
    diff_targets_dict = sorted(diff_targets_dict.items(), key=lambda x: x[1], reverse=True)
    return diff_targets_dict


def MSE(a, b):
    """
    description: calculates mean squared error
    :param a: np.array of shape (n,)
    :param b: np.array of shape (n,)
    :return:
        MSE
    """
    assert len(a) == len(b)
    return sum((a - b) ** 2) / (2 * len(a))


def get_new_marker_combo(best_marker_combos, combos_so_far):
    """
    determine new marker combination
    :param best_marker_combos: list of tuples of the form [('marker_name1',MSE1),('marker_name2',MSE2),...]
    :param combos_so_far: list of lists of marker combos already used [['markerA','markerB'],['markerC','markerD'],...]
    :return:
        list of names of choses markers (2 entries) of form ['marker1','marker2']
    """
    n = len(best_marker_combos)
    for m1 in range(n):
        finished = 0
        cand_marker1 = best_marker_combos[m1][0]
        for m2 in range(m1 + 1, n):
            cand_marker2 = best_marker_combos[m2][0]
            if not (([cand_marker1, cand_marker2] in combos_so_far) or ([cand_marker2, cand_marker1] in combos_so_far)):
                finished = 1
                break
        if finished == 1:
            break
    assert finished == 1
    return [cand_marker1, cand_marker2]


def fraction_targets_non_targets(
    filtered_obs, cluster_string, cluster_number,
):
    """
    determine fraction targets non-targets and choose weight for loss function accordingly
    :param filtered_obs: dataframe with all markers and filtered cells up to now
    :param cluster_string: string name of column of cluster algorithm, e.g. 'louvain' or 'leiden'
    :param cluster_number: string name of target cluster
    :return:
        weight: double, weight parameter for loss function
    """

    # To Do: add check whether cluster_string is a column of filtered obs
    nr_targets = sum(filtered_obs[cluster_string] == cluster_number)
    nr_non_targets = sum(filtered_obs[cluster_string] != cluster_number)
    assert nr_targets > 0
    weight = np.log2(nr_non_targets / nr_targets)
    # weight will later on be of the form 2^
    return weight


# DEPRECATED
def fraction_targets_vanilla(filtered_obs, cluster_string):  # To Do: remove as this is a duplicated function from above
    """
    determine fraction targets non-targets and choose weight for loss function accordingly
    :param filtered_obs: dataframe with all markers and filtered cells up to now
    :param cluster_string: string name of column of cluster algorithm, e.g 'louvain' or 'leiden'
    :return:
        weight: double, weight parameter for loss function
    """
    nr_targets = sum(filtered_obs[cluster_string] == 1)
    nr_non_targets = sum(filtered_obs[cluster_string] != 1)
    assert nr_targets > 0
    weight = np.log2(nr_non_targets / nr_targets)
    # weight will later on be of the form 2^
    return weight


def get_penalty_distance(normal_vectors, biases, mid_point_targets):
    """
    penalty term for hyperplanes that lie far away from the center of the target points
    :param normal_vectors: list of torch.tensor of torch.Size([2])
    :param biases: list of torch.tensor of torch.Size([1])
    :param mid_point_targets: torch.tensor of torch.Size([2])
    :return:
        torch.tensor of torch.Size([1]) penalty value
    """
    penalty_distance = torch.tensor([0.0])
    for j in range(len(normal_vectors)):
        penalty_distance += torch.abs(normal_vectors[j] @ mid_point_targets.float() + biases[j]) / torch.norm(
            normal_vectors[j]
        )
    return penalty_distance / len(normal_vectors)


def get_mean_target_value(target_df):
    """
    examine mean location of target population
    :param target_df: pandas data frame output of create_target_df
    :return:
        torch.tensor of torch.Size([2]) mean location of target cells
    """
    only_targets_df = target_df[target_df["label"] == 1]
    return torch.tensor(np.mean(only_targets_df.values[:, 0:2], axis=0)).float()


# normal_vectors
def norm1_normal_vectors(normal_vectors):
    """
    normalize normal vectors
    :param normal_vectors: list of torch.tensor of torch.Size([2])
    :return:
        list of torch.tensor of torch.Size([2]) - normalized normal_vectors
    """
    normal_vectors_new = []
    for j in range(len(normal_vectors)):
        normal_vectors_new.append(normal_vectors[j] / torch.norm(normal_vectors[j]))
    return normal_vectors_new


def sum_projections(normal_vectors):
    """
    :param description: penalty term for collinearity of normal vectors
    :param normal_vectors: list of torch.tensor of torch.Size([2]) (normalized)
    :return:
        sum_projection: torch.tensor of torch.Size([2]) penalty term for collinearity
    """
    nr_norm_vectors = len(normal_vectors)
    sum_projection = 0
    for u in range(nr_norm_vectors):
        for s in range(u + 1, nr_norm_vectors):
            sum_projection += torch.abs(normal_vectors[u] @ normal_vectors[s])
    return sum_projection


def return_f1_values(summary_gating_dicts):
    """
    return f1 for each hierarchy
    :param summary_gating_dicts: dictionary where each element is a gating dict,
                                 e.g. key '1' contains gating_dict of hierarchy 1
                                      key '2' contains gating_dict of hierarchy 2
    :return:
        np.array of shape (len(summary_gating_dicts),)
    """
    f1_dict = []
    for j in range(len(summary_gating_dicts)):
        f1_dict.append(summary_gating_dicts[str(j + 1)]["f1"])
    return np.array(f1_dict)


def penalty_1_99_targets(target_df, normal_vectors, biases, nr_hyperplanes):
    """
    calculates penalty term forcing hyperplanes to be close to points that
    lie between the 1 and 99 percentile
    :param target_df: panda data frame output of create_target_df
    :param normal_vectors: list of torch.tensor of torch.Size([2])
    :param biases: list of torch.tensor of torch.Size([1])
    :param nr_hyperplanes: int
    :return:
        torch.tensor of torch.Size([1]) penalty term forcing hyperplanes
        to lie close to most of points
    """
    only_targets_df = target_df[target_df["label"] == 1]
    perc1_1 = np.percentile(only_targets_df.values[:, 0], 1)
    perc1_2 = np.percentile(only_targets_df.values[:, 1], 1)
    perc99_1 = np.percentile(only_targets_df.values[:, 0], 99)
    perc99_2 = np.percentile(only_targets_df.values[:, 1], 99)
    only_target_filtered = only_targets_df[
        (only_targets_df.values[:, 0] > perc1_1)
        & (only_targets_df.values[:, 0] < perc99_1)
        & (only_targets_df.values[:, 1] > perc1_2)
        & (only_targets_df.values[:, 1] < perc99_2)
    ]
    dist_points = 0.0
    for j in range(nr_hyperplanes):
        dist_points += torch.abs(
            torch.mean(normal_vectors[j] @ torch.tensor(only_target_filtered.values[:, 0:2]).float().T + biases[j])
        )
    return (1 / nr_hyperplanes) * dist_points


def only_f1(target_df_dicts, summary_gating_dicts, nr_max_hierarchies, first_hierarchy=1):
    """
    prints hierarchy performance summary
    :param target_df_dicts: dictionary, where target_df_dicts[hierarchy_nr] equals targets dict of that hierarchy
    :param summary_gating_dicts: dictionary, where summary_gating_dicts[hierarchy_nr] equals gating dict of that hierarchy
    :param nr_max_hierarchies: int, maximum number of hierarchies that should be plotted
    :param first_hierarchy: int, first hierarchy to print results for
    :return:
        print of summary performance
    """
    f1_total_list = []
    for hierarchy in range(1, nr_max_hierarchies + 1):
        f1 = print_results_current_pos_only_f1(summary_gating_dicts, first_hierarchy, hierarchy)
        f1_total_list.append(f1)
        # visualizing_gate(target_df_dicts,summary_gating_dicts,hierarchy)
    return f1_total_list


def return_best_marker_combo_tree(filtered_obs, channels):
    X_tree = filtered_obs[channels].values
    y_tree = filtered_obs["label"].values
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_tree, y_tree)
    s = export_graphviz(clf)
    res_matrix = np.zeros([len(s.split('[label="X')[1:]), 2])
    for j in range(len(s.split('[label="X')[1:])):
        res_matrix[j, 0] = int(s.split('[label="X')[1:][j][1:2])
        try:
            res_matrix[j, 1] = float(s.split("ngini = ")[1:][j][0:5])
        except Exception:
            try:
                res_matrix[j, 1] = float(s.split("ngini = ")[1:][j][0:4])
            except Exception:
                res_matrix[j, 1] = float(s.split("ngini = ")[1:][j][0:3])
    channels_so_far = []
    output = []
    for q in range(len(s.split('[label="X')[1:])):
        if channels[int(res_matrix[q, 0])] not in channels_so_far:
            output.append((channels[int(res_matrix[q, 0])], res_matrix[q, 1]))
            channels_so_far.append(channels[int(res_matrix[q, 0])])
    return output


def renorm(min_val, max_val, y):
    """
    renormalize data
    :param min_val: double
    :param max_val: double
    :param y: double -> value to renormalize
    """
    return y * (max_val - min_val) + min_val


def renorm_hierarchy(re_df, marker, norm_values):
    """
    renormalizes output
    :param renorm_df: data frame with min and max value used for transformation
    :param marker string, marker name
    :param norm_values: array, values to normalize
    :output array, renormalized
    """
    min_marker = re_df[marker].loc["min"]
    max_marker = re_df[marker].loc["max"]
    renorm_values = renorm(min_marker, max_marker, norm_values)
    return renorm_values


def renorm_gating_dict_hierarchy(gating_dict, key, hierarch, re_df):
    """
    renormalizes gating_dict output for specific gating hierarchy
    :param gating_dict : dictionary, output of function do_complete_gating
    :param key int, identifier
    :param hierarchy str, hierarchy
    :returns
        renorm_gate_points  array, renromalized edge points
        edge_list list, edges to connect

    """
    option = 1
    marker1 = gating_dict[key][option][hierarch]["marker_combo"][0]
    marker2 = gating_dict[key][option][hierarch]["marker_combo"][1]
    p1 = np.array(gating_dict[key][option][hierarch]["gate_points"])[:, 0]
    p2 = np.array(gating_dict[key][option][hierarch]["gate_points"])[:, 1]
    ren_corner1 = renorm_hierarchy(re_df, marker1, p1)
    ren_corner2 = renorm_hierarchy(re_df, marker2, p2)
    renorm_gate_points = np.array([ren_corner1, ren_corner2]).T
    edge_list = []
    for q in range(len(gating_dict[key][option][hierarch]["gate_edges"])):
        x1 = np.array(gating_dict[key][option][hierarch]["gate_edges"][q])[:, 0]
        x2 = np.array(gating_dict[key][option][hierarch]["gate_edges"][q])[:, 1]
        ren_ed1 = renorm_hierarchy(re_df, marker1, x1)
        ren_ed2 = renorm_hierarchy(re_df, marker2, x2)
        edge_list.append(list(np.array([ren_ed1, ren_ed2]).T))
    ren_marker_1 = renorm_hierarchy(re_df, marker1, gating_dict[key][0][hierarch][marker1].values)
    ren_marker_2 = renorm_hierarchy(re_df, marker1, gating_dict[key][0][hierarch][marker2].values)
    lab = gating_dict[key][0][hierarch]["label"]
    ID = gating_dict[key][0][hierarch]["cell_ID"]
    df_ren_cells = pd.DataFrame(
        np.array([list(ren_marker_1), list(ren_marker_2), list(lab), list(ID)]).T,
        columns=[marker1, marker2, "label", "cell_ID"],
    )
    return renorm_gate_points, edge_list, df_ren_cells


def return_best_marker_combo_single_tree(filtered_obs, channels):
    """
    return a dictionary which shows which markers can be separated by a simple Decision Tree with highest F1 score
    :param filtered_obs: panda dataframe
    :param channels: list of available channels
    :return:
        diff_tree_dict: sorted dictionary with (key:name_of_marker, value:F1 score)
                       first entry marker with highest F1 score
    """
    DTC = DecisionTreeClassifier(max_depth=4)
    y_true = filtered_obs["label"].values
    tree_dict = {}
    for chan in channels:
        X_in = filtered_obs[chan].values.reshape(-1, 1)
        DTC.fit(X_in, y_true)
        y_pred = DTC.predict(X_in)
        tree_dict[chan] = f1_score(y_true, y_pred)
    diff_tree_dict = sorted(tree_dict.items(), key=lambda x: x[1], reverse=True)
    return diff_tree_dict


def return_best_marker_combo_single_svm(filtered_obs, channels):
    """
    return a dictionary which shows which markers can be separated by a linear SVM with highest F1 score
    :param filtered_obs: panda dataframe
    :param channels: list of available channels
    :return:
        diff_svm_dict: sorted dictionary with (key:name_of_marker, value:F1 score)
                       first entry marker with highest F1 score
    """
    clf = LinearSVC(class_weight="balanced")
    y_true = filtered_obs["label"].values
    svm_dict = {}
    for chan in channels:
        X_in = filtered_obs[chan].values.reshape(-1, 1)
        clf.fit(X_in, y_true)
        y_pred = clf.predict(X_in)
        svm_dict[chan] = f1_score(y_true, y_pred)
    diff_svm_dict = sorted(svm_dict.items(), key=lambda x: x[1], reverse=True)
    return diff_svm_dict


def create_final_output(cell_data, key, re_gating_dict, overview, channels, res_in_gate, cluster_string="louvain"):
    """


    Parameters
    ----------
    cell_data : TYPE
        DESCRIPTION.
    key : TYPE
        DESCRIPTION.
    re_gating_dict : TYPE
        DESCRIPTION.
    overview : TYPE
        DESCRIPTION.
    channels : TYPE
        DESCRIPTION.
    res_in_gate : TYPE
        DESCRIPTION.
    cluster_string : str
        DESCRIPTION.

    Returns
    -------
    overview_df : TYPE
        DESCRIPTION.

    """
    nr_total_hierarch = len(re_gating_dict[key][0])
    idx = overview["key"] == key
    # best_hierarch = overview[idx]['best_hierarchy'].values[0]
    clust = overview[idx]["cluster_number"].values[0]
    true_label = (cell_data[cluster_string] == clust).values * 1
    overview_df = cell_data[channels]
    overview_df["true_label"] = true_label

    for hier in range(1, nr_total_hierarch + 1):
        df_best_hierarch = re_gating_dict[key][0][str(hier)]
        df_best_hierarch["final_pred"] = res_in_gate[key][str(hier)]
        cell_ID_gate = list(df_best_hierarch[df_best_hierarch["final_pred"] == 1]["cell_ID"])
        one_hot = np.zeros(len(cell_data))
        for idx in cell_ID_gate:
            one_hot[int(idx)] = 1
            overview_df["gate_" + str(hier)] = one_hot
    return overview_df


def preprocess_adata_gating(adata, cluster_string):
    """


    Parameters
    ----------
    adata : AnnData object
        complete input Data for gating tool.
    cluster_string : string
        name of column with cluster identifier in adata.obs (e.g 'louvain')

    Returns
    -------
    cell_data : panda DataFrame
        preprocessed DataFrame for gating model

    """
    cell_data = adata_to_df_gating(adata, cluster_string)
    cell_identifier = list(cell_data.index)
    cell_data["cell_ID"] = cell_identifier
    return cell_data


def prepare_df(general, hierarch, re_gating_dict, key):
    index_hierarch = list(re_gating_dict[key][0][hierarch]["cell_ID"])
    current_gate = "gate_" + hierarch
    markers = list(re_gating_dict[key][0][hierarch].columns[0:2].values)
    columns = markers + ["true_label"] + [current_gate]
    current_gate_df = general.loc[index_hierarch][columns]
    current_gate_df.rename(columns={current_gate: "gate"}, inplace=True)
    gate_points = current_gate_df[current_gate_df["gate"] == 1].iloc[:, 0:2].values
    hull = ConvexHull(gate_points)
    hull_path = Path(gate_points[hull.vertices])
    targets = current_gate_df[current_gate_df["true_label"] == 1].iloc[:, 0:2].values
    non_targets = current_gate_df[current_gate_df["true_label"] == 0].iloc[:, 0:2].values
    current_gate_df["cell_ID"] = index_hierarch
    return current_gate_df, gate_points, hull, hull_path, targets, non_targets, gate_points


def get_nr_hierarch_general(general):
    h = 1
    terminated = False
    while not terminated:
        if "gate_" + str(h) in general.columns:
            h += 1
        else:
            terminated = True
    nr_hierarch_general = h - 1
    return nr_hierarch_general


def apply_convex_hull(general, key, re_gating_dict):
    n_h = get_nr_hierarch_general(general)
    hull_dict = {}
    path_dict = {}
    current_gate_dicts = {}
    gate_points_dict = {}
    for hierarch in range(1, n_h + 1):
        current_gate_df, gate_points, hull, hull_path, targets, non_targets, gate_points = prepare_df(
            general, str(hierarch), re_gating_dict, key
        )
        hull_dict[hierarch] = hull
        path_dict[hierarch] = hull_path
        current_gate_dicts[hierarch] = current_gate_df
        gate_points_dict[hierarch] = gate_points
        append_general(general, hierarch, path_dict, current_gate_dicts)
    make_final_gate(general, n_h)
    return n_h, hull_dict, path_dict, current_gate_dicts, gate_points_dict


def make_final_gate(general, n_h):
    general["final_gate_0"] = np.ones(len(general))
    survived = general["new_gate_1"].values
    for u in range(1, n_h + 1):
        general["final_gate_" + str(u)] = survived.copy()
        if u < n_h:
            survived *= general["new_gate_" + str(u + 1)]


def append_general(general, hierarch, hull_path_dict_hierarchy, current_gate_dicts):
    markers = current_gate_dicts[hierarch].iloc[:, 0:2].columns
    new_gate_vec = []
    for j in range(len(general)):
        new_gate_vec.append(hull_path_dict_hierarchy[hierarch].contains_point(general[markers].values[j, :]) * 1)
    general["new_gate_" + str(hierarch)] = new_gate_vec.copy()


def metrics_new(clust_string, general, n_h, show_metrics_df=True, save_metrics_df=True, save_path=os.getcwd()):
    """
    Visualization of gating strategy via heatmaps (non-target population).

    Parameters
    ----------
    clust_string : string
        an identifier for the current cluster, e.g. '4'
    general : pd.DataFrame
        dataframe containing all relevant infos for visualization
        output of function 'process_results'
    n_h : int
        number of hierarchies in best gating strategy
    save_metrics_df : True or False (default True)
        whether to print DataFrame for performance overview on console
    save_metrics_df : True or False (default True)
        whether to save DataFrame for performance overview
    save_path : str (default os.getcwd() -> current working directory)
        path (location) to save DataFrame
    """

    f1_rec_prec_out = pd.DataFrame(index=["f1", "recall", "precision"])
    for hier in range(1, n_h + 1):
        g_string = "final_gate_" + str(hier)
        h_string = "hierarchy_" + str(hier)
        f, r, p = (
            np.round(f1_score(general["true_label"].values, general[g_string].values), 6),
            np.round(recall_score(general["true_label"].values, general[g_string].values), 6),
            np.round(precision_score(general["true_label"].values, general[g_string].values), 6),
        )
        f1_rec_prec_out[h_string] = f, r, p
    if save_metrics_df:
        path_out = os.path.join(save_path, "cluster_" + clust_string)
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        save_location = os.path.join(path_out, "performance.csv")
        f1_rec_prec_out.to_csv(save_location)
    if show_metrics_df:
        print(f1_rec_prec_out)
    return f1_rec_prec_out


def renormalize_gating_output(gating_dict, re_df):
    """
    renormalizes output of gating model
    :param gating_dict dictionary, output of do_complete_gating
    :returns renorm_gating_dict, dict with renormalization
    """
    renorm_gating_dict = {}

    for key in gating_dict.keys():
        cell_dict = {}
        gate_dict = {}

        for hchy in gating_dict[key][0].keys():
            curr_dict = {}
            gate_points, edge_list, cell_val = renorm_gating_dict_hierarchy(gating_dict, key, str(hchy), re_df[key])
            cell_dict[str(hchy)] = cell_val.copy()
            curr_dict["marker_combo"] = gating_dict[key][1][str(hchy)]["marker_combo"]
            curr_dict["gate_points"] = gate_points.copy()
            curr_dict["gate_edges"] = edge_list.copy()
            gate_dict[str(hchy)] = curr_dict.copy()
        renorm_gating_dict[key] = [cell_dict, gate_dict]
    return renorm_gating_dict


def process_results(cell_data, overview, results_dictionaries, renorm_df_dict, res_in_gates, channels, cluster_string):
    general_dict = {}
    clust_string_dict = {}
    re_gating_dict = renormalize_gating_output(results_dictionaries, renorm_df_dict)
    for key in list(overview["key"]):
        clust_string = overview[overview["key"] == key]["cluster_number"].values[0]
        general = create_final_output(
            cell_data, key, re_gating_dict, overview, channels, res_in_gates, cluster_string=cluster_string
        )
        general_dict[key] = general
        clust_string_dict[key] = clust_string
    return clust_string_dict, re_gating_dict, general_dict

def add_tight_hull_hierarchy(base_df,hierarchy,meta_info,run_ID):
    #base_df = meta_info['general_summary'][run_ID]
    #run_ID - z.B 0
    #meta_info -> .npy output
    #run_ID -> e.g. 0
    marker = meta_info['gating_summary'][run_ID][1][hierarchy]['marker_combo']
    marker1 = marker[0]
    marker2 = marker[1]
    base_df['gate_hull_0'] = base_df['final_gate_0'].copy().astype(int)
    targets_in_gate = base_df[(base_df['final_gate_' +hierarchy] == 1)&(base_df['true_label'] == 1)][[marker1,marker2]].values
    hull = ConvexHull(targets_in_gate)
    edge_points_indices = hull.vertices
    hull_path = Path(targets_in_gate[hull.vertices])
    final_hull = [hull_path.contains_point(point) for point in base_df[[marker1,marker2]].values]*1
    #account for numeric error in .contains method
    loc_numeric_hull_error = (base_df['final_gate_' +hierarchy] == True)&(base_df['true_label'] == True)*(np.array(final_hull) == False)
    final_hull += loc_numeric_hull_error
    base_df['hull_' + hierarchy] = [int(val) for val in final_hull]
    base_df['hull_' + hierarchy] = [int(val) for val in final_hull]
    return base_df

def add_gate_tight(base_df,meta_info,run_ID):
    column_numbers = [int(re.findall(r'^final_gate_(\d+)$', col)[0]) for col in base_df.columns if re.match(r'^final_gate_\d+$', col)]

    total_hierarchies = np.max(column_numbers)

    for h in range(1,total_hierarchies+1):
        base_df = add_tight_hull_hierarchy(base_df,str(h),meta_info = meta_info,run_ID=run_ID)

    base_df['gate_hull_1'] = base_df['hull_1'].copy()
    if total_hierarchies>1:
        for hc in range(2,total_hierarchies+1):
            base_df['gate_hull_' +str(hc)] = base_df['gate_hull_' +str(hc -1)]*base_df['hull_'+str(hc)]
    return base_df

def add_tight_metric(base_df,path_to_save_performance):
    column_numbers = [int(re.findall(r'^final_gate_(\d+)$', col)[0]) for col in base_df.columns if re.match(r'^final_gate_\d+$', col)]
    total_hierarchies = np.max(column_numbers)
    f1_rec_prec_out = pd.DataFrame(index=["f1", "recall", "precision"])
    for hier in range(1, total_hierarchies + 1):
        g_string = "gate_hull_" + str(hier)
        h_string = "hierarchy_" + str(hier)
        f, r, p = (
            np.round(f1_score(base_df["true_label"].values, base_df[g_string].values), 6),
            np.round(recall_score(base_df["true_label"].values, base_df[g_string].values), 6),
            np.round(precision_score(base_df["true_label"].values, base_df[g_string].values), 6),
        )
        f1_rec_prec_out[h_string] = f, r, p
        
    save_location = os.path.join(path_to_save_performance, "performance.csv")
    f1_rec_prec_out.to_csv(save_location)
    
def add_visualization_hierarchy(meta_info,base_df,save_location,hierarchy,run_ID,
                                gate_color = 'tab:red',
                                targets_color = '#FF4F00',
                                non_targets_color = '#2d5380',
                                save = True,
                                show=True):
    population = meta_info['clusterkeys'][run_ID]
    marker = meta_info['gating_summary'][run_ID][1][hierarchy]['marker_combo']
    marker1 = marker[0]
    marker2 = marker[1]

    points_df = base_df[base_df['gate_hull_' + str(int(hierarchy)-1)] == 1]


    in_gate_target = base_df[(base_df['gate_hull_' + hierarchy]==1)&(base_df['true_label']==1)]


    in_gate_target_points = in_gate_target[[marker1,marker2]].values

    hull = ConvexHull(in_gate_target_points)

    edge_points_indices = hull.vertices

    edge_points = in_gate_target_points[edge_points_indices]

    color_map = {0: non_targets_color, 1: targets_color}



    plt.figure(figsize=(6, 4))
    plt.scatter(points_df[marker1].values, points_df[marker2].values, c=points_df['true_label'].map(color_map),alpha = 0.4,s=11)
    for simplex in hull.simplices:
        plt.plot(in_gate_target_points[simplex, 0], in_gate_target_points[simplex, 1], color=gate_color,alpha = 0.8,linewidth = 3)
    plt.tight_layout()
    plt.xlabel(marker1,fontsize = 11)
    plt.ylabel(marker2,fontsize = 11)
    plt.title('cluster ' + population +' - hierarchy ' + hierarchy,fontsize = 12)
    if save:
        save_ID = 'cluster_' + population + '_hierarchy_' + hierarchy + '.pdf'
        save_path = os.path.join(save_location, save_ID)
        plt.savefig(save_path,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return edge_points

def add_tight_analysis(meta_info,run_ID,save_loc):
    population = meta_info['clusterkeys'][run_ID]
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)

    base_df = meta_info['general_summary'][run_ID]
    base_df = add_gate_tight(base_df,meta_info,run_ID)
    add_tight_metric(base_df = base_df,
                     path_to_save_performance = save_loc)



    column_numbers = [int(re.findall(r'^final_gate_(\d+)$', col)[0]) for col in base_df.columns if re.match(r'^final_gate_\d+$', col)]
    total_hierarchies = np.max(column_numbers)
    for hierarchy in range(1,total_hierarchies+1):
        edge_points = add_visualization_hierarchy(meta_info = meta_info,
                                base_df = base_df,
                                save_location = save_loc,
                                hierarchy = str(hierarchy),
                               run_ID = run_ID)
        edge_save_loc = os.path.join(save_loc,'cluster_' + population + '_gate_edges_hierarchy_' + str(hierarchy) + '.csv')
        pd.DataFrame(edge_points,columns = ['x_coordinate','y_coordinate']).to_csv(edge_save_loc)
        
def plot_metric_tight(meta_info,run_ID,save_loc,save=True,show=True):
    #save_loc -> path to where 'performance.csv' is saved
    #color Java palette: recall: #e2998a, precision: #0c7156, recall: #e2998a
    population = meta_info['clusterkeys'][run_ID]
    scores = pd.read_csv(os.path.join(save_loc,'performance.csv'),index_col = 0)
    f1_val = scores.T["f1"]
    recall_val = scores.T["recall"]
    precision_val = scores.T["precision"]
    y_names = list(scores.T.index)
    plt.figure()
    plt.plot(y_names, recall_val, "-o", color="#e2998a", label="recall", linestyle="dashdot")
    plt.plot(y_names, f1_val, "-o", color="#663171", label="F1", linestyle="dashdot")
    plt.plot(y_names, precision_val, "-o", color="#0c7156", label="precision", linestyle="dashdot")
    plt.legend()
    plt.title("Performance - cluster " + population  , fontsize=12)
    plt.ylim([0, 1.01])
    plt.xlabel("Gating Depth", fontsize=11)
    if save:
            save_ID = 'cluster_' + population + '_performance_graphic.pdf'
            save_path = os.path.join(save_loc, save_ID)
            plt.savefig(save_path,bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def get_f1_hierarch(perf_loc):
    tab = pd.read_csv(perf_loc)
    #tab = pd.read_csv('level_' +str(level) + '/cluster_' + celltype + '/performance.csv')
    best_loc_f1 = np.argmax(tab.iloc[0][1:])
    f1 = tab.iloc[0][1:][best_loc_f1]
    recall = tab.iloc[1][1:][best_loc_f1]
    precision = tab.iloc[2][1:][best_loc_f1]
    hierarchy = best_loc_f1 + 1
    return f1,recall,precision,hierarchy

def make_performance_summary(meta_info_path,target_location):
    meta_info = np.load(meta_info_path,allow_pickle=True).item()
    run_IDs = len(meta_info['clusterkeys'])
    df = pd.DataFrame(columns =['cluster','f1','recall','precision','hierarchy'])
    for run_ID in range(run_IDs):
        ident = meta_info['clusterkeys'][run_ID]
        cluster_name = 'cluster_' + ident
        perf_loc = os.path.join(target_location,cluster_name,'performance.csv')
        df = df.append(pd.DataFrame([[ident] + list(get_f1_hierarch(perf_loc))],columns=['cluster','f1','recall','precision','hierarchy']))
    df.to_csv(os.path.join(target_location,'performance_summary.csv'),index=False)

def make_marker_summary(meta_info_path,target_location):
    meta_info = np.load(meta_info_path,allow_pickle=True).item()
    run_IDs = len(meta_info['clusterkeys'])
    df = pd.DataFrame(columns =['marker','hierarchy','cluster'])
    for run_ID in range(run_IDs):
        ident = meta_info['clusterkeys'][run_ID]
        cluster_name = 'cluster_' + ident
        perf_loc = os.path.join(target_location,cluster_name,'performance.csv')
        (_,_,_,max_hierarchy) = get_f1_hierarch(perf_loc)
        for hierarchy in range(1,max_hierarchy+1):
            df = df.append(return_marker_combo_df(meta_info,run_ID,str(hierarchy)))
    df.to_csv(os.path.join(target_location,'marker_summary.csv'),index=False)

def return_marker_combo_df(meta_info,run_ID,hierarchy):
    marker = meta_info['gating_summary'][run_ID][1][hierarchy]['marker_combo']
    marker_df = pd.DataFrame(marker,columns = ['marker'])
    marker_df['hierarchy'] = hierarchy
    marker_df['cluster'] = meta_info['clusterkeys'][run_ID]
    return marker_df

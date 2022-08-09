#!/usr/bin/env python
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def simulate_complete_FACS(cell_data, cluster_name):
    """
    Simulate an entire FACS dataset from a real measurement,
    with the same cluster sizes as the original data set

    Parameters
    ----------
    cell_data : pandas DataFrame
        Expression values of a flow data set including metadata and cluster
        information.
    cluster_name : str
        column in cell_data, denotes the different populations.

    Returns
    -------
    new_df : pandas DataFrame
        simulated FACS measurements.

    """

    population = list(pd.unique(cell_data[cluster_name]))
    new_df = pd.DataFrame(columns=cell_data.columns)
    for pop in population:
        cell_data_pop = cell_data[cell_data[cluster_name] == pop]
        nr_of_cells = len(cell_data_pop)
        df_pop = simulate_FACS_per_population(cell_data_pop, cluster_name, pop, nr_of_cells)
        new_df = new_df.append(df_pop)
    return new_df


def simulate_FACS_per_population(cell_data, cluster_name, cluster_number, nr_of_cells):
    """
    Simulate a single population of cells from a FACS measurement

    Parameters
    ----------
    cell_data : pandas DataFrame
        (adata.X, columns = list(adata.var.index)) only one population.
    cluster_name : str
        cluster name, e.g. 'leiden'.
    cluster_number : str
        valid name in the column <cluster_name>, e.g. '0'.
    nr_of_cells : int
        desired number of sampled cells from that population.

    Returns
    -------
    df_sampled_cl : pandas DataFrame
        DataFrame of simulated cells.

    """

    nr_markers = len(cell_data.columns) - 1
    markers = list(cell_data.columns[:nr_markers])
    cov = np.cov(cell_data.values[:, 0:nr_markers].astype(float).T)
    mean = np.mean(cell_data.values[:, 0:nr_markers].astype(float).T, axis=1)
    sampled_cl = np.random.multivariate_normal(mean, cov, nr_of_cells)
    df_sampled_cl = pd.DataFrame(sampled_cl, columns=markers)
    cluster = [cluster_number] * nr_of_cells
    df_sampled_cl[cluster_name] = cluster
    return df_sampled_cl


def simulate_complete_FACS_variable(cell_data, cluster_name, cluster_to_simulate, nr_of_cells_to_simulate):
    """
    Simulate an entire FACS dataset from a real measurement,
    with a variable number of cells

    Parameters
    ----------
    cell_data : pandas DataFrame
        (adata.X, columns = list(adata.var.index)) only one population.
    cluster_name : str
        cluster name, e.g. 'leiden'.
    cluster_to_simulate : list
        e.g. ['1','2','5'].
    nr_of_cells_to_simulate : list
        same length as cluster_to_simulate, e.g. [1000,5000,100000].

    Returns
    -------
    new_df : pandas DataFrame
        DataFrame of simulated cells.

    """

    new_df = pd.DataFrame(columns=cell_data.columns)
    for pop in range(len(cluster_to_simulate)):
        cell_data_pop = cell_data[cell_data[cluster_name] == cluster_to_simulate[pop]]
        nr_of_cells = nr_of_cells_to_simulate[pop]
        df_pop = simulate_FACS_per_population(cell_data_pop, cluster_name, cluster_to_simulate[pop], nr_of_cells)
        new_df = new_df.append(df_pop)
    return new_df

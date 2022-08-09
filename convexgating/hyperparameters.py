"""
Script for specifying desired CLUSTERS and MODEL HYPERPARAMETERS
"""
# import os

nr_max_hierarchies = 5  # max number of hierarchies in gating strategy
PC = True  # use PCA for gate initialization
learning_rate = 0.05  # learning rate in SGD
iterations = 50  # iterations in SGD
batch_size = 10000  # batch size in SGD
nr_hyperplanes = 8  # number of hyperplanes to use
grid_divisor = 3  # hyperparameter in adaptive grid search
refinement_grid_search = 2  # hyperparameter in adaptive grid search
weight_version = 1  # weight version (1 recommended)
marker_sel = "heuristic"  # marker selection method ('tree','svm','heuristic')
arange_init = [0, 2, 4, 8, 10]  # initialization adaptive grid search
show_HEAT = True  # True or False - show heat plots
save_HEAT = True  # True or False - save heat plots
show_SCATTER = True  # True or False - show scatter plots
save_SCATTER = True  # True or False - save scatter plots
show_metrics_df = True  # True or False - show metrics overview dataframe
save_metrics_df = True  # True or False - save metrics overview dataframe
save_metrics_plot = True  # True or False - show metrics overview plot
show_metrics_plot = True  # True or False - save metrics overview plot
# save_path= os.getcwd()            #string, path to desired save location

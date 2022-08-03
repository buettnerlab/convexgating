#!/usr/bin/env python
# import convexgating
# from convexgating import helper, plotting, simulation, tools

__all__ = [
    "adata_to_df_gating",
    "do_adaptive_grid_search",
    "do_complete_gating",
    "find_2D_gate",
    "test_without_learning",
    "renormalize_gating_output",
    "simulate_FACS_per_population",
    "simulate_complete_FACS",
    "simulate_complete_FACS_variable",
    "GradientDescentMulti",
    "MSE",
    "adapt_init_norm_bias",
    "classify_points",
    "create_target_df",
    "fraction_targets_non_targets",
    "fraction_targets_vanilla",
    "generate2unit_planes",
    "get_candidate_points",
    "get_mean_target_value",
    "get_new_marker_combo",
    "get_normal_v_biases",
    "get_penalty_distance",
    "get_points_to_connect",
    "get_relevant_points",
    "get_two_points_each_hyperplane",
    "heuristic_markers",
    "initialize_norm_bias",
    "line_intersect",
    "marker_greedy_summary",
    "norm1_normal_vectors",
    "normalization",
    "normalize_column",
    "only_f1",
    "penalty_1_99_targets",
    "possible_marker_combinations",
    "print_results_current_pos_only_f1",
    "renormalize_column",
    "return_best_marker_combo",
    "return_best_marker_combo_tree",
    "create_final_output",
    "return_f1_values",
    "sigmoid_s",
    "sigmoid_s_torch",
    "sum_projections",
    "target_summary",
    "summary_metrics",
    "trange",
    "two_points_hyperplane2D",
    "value_halfspace",
    "print_results_current_pos",
    "show_gating_strategy",
    "visualize_heat",
    "show_heat_strategy",
    "visualize_final_gate",
    "final_results_visualization",
    "visualizing_gate",
    "plot_F1_recall_precision",
    "vizualize",
    "visualize_gate_hierachy",
    "renorm_gating_dict_hierarchy",
    "renorm_hierarchy",
    "renorm",
    "return_best_marker_combo_single_tree",
    "return_best_marker_combo_single_svm",
    "save_score_overview",
    "preprocess_adata_gating",
    "do_gating",
    "prepare_df",
    "get_nr_hierarch_general",
    "apply_convex_hull",
    "make_final_gate",
    "append_general",
    "metrics_new",
    "do_SCATTER",
    "do_HEAT_targets",
    "do_HEAT_non_targets",
    "do_plot_metrics",
    "generate_output",
    "FIND_GATING_STRATEGY",
    "process_results",
]

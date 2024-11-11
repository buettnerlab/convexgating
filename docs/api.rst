
===
API
===


Usage Example -  Single Target Population
-------------------
Generating gating strategy for cells or events labeled 'cluster_A' from an AnnData object (adata) using labels found in adata.obs[cluster_string].

.. code-block:: console
    
    import convexgating as cg
    import scanpy as sc
    
    adata = sc.read_h5ad(adata_path)
    cluster_string = 'clustering'                                       #column in adata.obs
    target_cluster = 'cluster_A'                                        #category in adata.obs[cluster_string]
    save = save_path
    cg.tools.CONVEX_GATING(adata=adata,
              cluster_numbers = [target_cluster],
              cluster_string = cluster_string,
              save_path=save)

Usage Example -  Multiple Target Population
-------------------
Generating a series of gating strategies for cells or events labeled 'cluster_A','cluster_B','cluster_C' from an AnnData object (adata) using labels found in adata.obs[cluster_string].

.. code-block:: console
    
    import convexgating as cg
    import scanpy as sc
    
    adata = sc.read_h5ad(adata_path)
    cluster_string = 'clustering'                                          #column in adata.obs
    target_clusters = ['cluster_A','cluster_B','cluster_C']                #categories in adata.obs[cluster_string]
    save = save_path
    cg.tools.CONVEX_GATING(adata=adata,
              cluster_numbers = target_clusters,
              cluster_string = cluster_string,
              save_path=save)
    


Core functions
---------------
.. autoclass:: convexgating.tools.CONVEX_GATING
.. autoclass:: convexgating.tools.gating_strategy

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='z', random_seed=2026, use_pca=True, n_comp=30):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.

    Parameters
    ----------
    use_pca : bool, default=True
        If True, run PCA on adata.obsm[used_obsm] before clustering.
        If False, cluster directly on adata.obsm[used_obsm].
    n_comp : int, default=30
        Number of PCA components to use when use_pca is True.
    """
    np.random.seed(random_seed)

    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter

    robjects.r('library(mclust)')
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']

    X = adata.obsm[used_obsm]

    # Convert sparse matrices and other array-like inputs to a NumPy array.
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    if use_pca:
        X = PCA(n_components=n_comp, random_state=2026).fit_transform(X)
        adata.obsm['z_pca_for_mclust'] = X

    # Use the local conversion context recommended for recent rpy2 versions.
    with localconverter(default_converter + numpy2ri.converter):
        r_X = robjects.conversion.py2rpy(X)

    res = rmclust(r_X, num_cluster, modelNames)

    # Retrieve classifications by name to avoid relying on their position.
    mclust_res = np.array(res.rx2('classification'))

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype(int)
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def refine_label(
    adata,
    n_neighbors=50,
    key='label',
    x_key='x_array',
    y_key='y_array',
    new_key=None,
    include_self=False
):
    """
    Smooth or refine labels by majority voting among spatial nearest neighbors.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with coordinate and label columns in adata.obs.
    n_neighbors : int, default=50
        Number of neighbors used for majority voting.
    key : str, default='label'
        Column in adata.obs containing the original labels.
    x_key : str, default='x_array'
        Column containing x coordinates.
    y_key : str, default='y_array'
        Column containing y coordinates.
    new_key : str or None, default=None
        If provided, write the refined labels to adata.obs[new_key].
        If None, return the labels without modifying adata.
    include_self : bool, default=False
        Whether to include the observation itself in the vote.

    Returns
    -------
    new_labels : np.ndarray
        Array of refined labels as strings.
    """
    # -------- 1. Validate inputs --------
    for col in [key, x_key, y_key]:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing column in adata.obs: {col}")

    labels = adata.obs[key].astype(str).to_numpy()
    coords = adata.obs[[x_key, y_key]].to_numpy(dtype=np.float64)

    n_cells = coords.shape[0]
    if n_cells == 0:
        raise ValueError("adata is empty and contains no observations.")

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be a positive integer.")

    # Query one extra neighbor when excluding self, which is usually the nearest.
    query_k = n_neighbors + 1 if not include_self else n_neighbors
    query_k = min(query_k, n_cells)

    # -------- 2. Set up the nearest-neighbor search --------
    nbrs = NearestNeighbors(n_neighbors=query_k, algorithm='auto', metric='euclidean')
    nbrs.fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)

    # -------- 3. Encode labels for efficient majority voting --------
    label_codes, unique_labels = pd.factorize(labels, sort=False)
    new_codes = np.empty(n_cells, dtype=label_codes.dtype)

    # -------- 4. Apply neighbor majority voting to each observation --------
    for i in range(n_cells):
        neigh_idx = indices[i]

        if not include_self:
            # Explicitly exclude self rather than assuming it is the first neighbor.
            neigh_idx = neigh_idx[neigh_idx != i]

        # Use up to n_neighbors, or all remaining neighbors for small samples.
        if len(neigh_idx) > n_neighbors:
            neigh_idx = neigh_idx[:n_neighbors]

        neigh_codes = label_codes[neigh_idx]

        # Count votes with NumPy's bincount for efficient majority voting.
        counts = np.bincount(neigh_codes)
        majority_code = counts.argmax()
        new_codes[i] = majority_code

    new_labels = unique_labels[new_codes].astype(str)

    # -------- 5. Optionally write the labels back to adata --------
    if new_key is not None:
        adata.obs[new_key] = new_labels

    return new_labels

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

    # 防止是稀疏矩阵或其他类型
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    if use_pca:
        X = PCA(n_components=n_comp, random_state=2026).fit_transform(X)
        adata.obsm['z_pca_for_mclust'] = X

    # 新版 rpy2 的推荐写法
    with localconverter(default_converter + numpy2ri.converter):
        r_X = robjects.conversion.py2rpy(X)

    res = rmclust(r_X, num_cluster, modelNames)

    # 更稳妥地按名字取 classification
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
    基于空间最近邻多数投票进行标签平滑/细化。

    Parameters
    ----------
    adata : AnnData
        输入的 AnnData 对象，adata.obs 中需要包含坐标列和标签列。
    n_neighbors : int, default=50
        用于多数投票的邻居数。
    key : str, default='label'
        原始标签所在的 adata.obs 列名。
    x_key : str, default='x_array'
        x 坐标列名。
    y_key : str, default='y_array'
        y 坐标列名。
    new_key : str or None, default=None
        如果提供，则把 refined label 写入 adata.obs[new_key]。
        如果为 None，则只返回结果，不写入 adata。
    include_self : bool, default=False
        投票时是否包含自己。

    Returns
    -------
    new_labels : np.ndarray
        细化后的标签数组（字符串类型）。
    """
    # -------- 1. 基本检查 --------
    for col in [key, x_key, y_key]:
        if col not in adata.obs.columns:
            raise ValueError(f"adata.obs 中缺少列: {col}")

    labels = adata.obs[key].astype(str).to_numpy()
    coords = adata.obs[[x_key, y_key]].to_numpy(dtype=np.float64)

    n_cells = coords.shape[0]
    if n_cells == 0:
        raise ValueError("adata 为空，没有观测点。")

    if n_neighbors <= 0:
        raise ValueError("n_neighbors 必须是正整数。")

    # 如果不包含自己，实际查询时需要多取一个邻居（因为最近的第一个通常是自己）
    query_k = n_neighbors + 1 if not include_self else n_neighbors
    query_k = min(query_k, n_cells)

    # -------- 2. 建立近邻搜索 --------
    nbrs = NearestNeighbors(n_neighbors=query_k, algorithm='auto', metric='euclidean')
    nbrs.fit(coords)
    indices = nbrs.kneighbors(coords, return_distance=False)

    # -------- 3. 编码标签，提高多数投票效率 --------
    label_codes, unique_labels = pd.factorize(labels, sort=False)
    new_codes = np.empty(n_cells, dtype=label_codes.dtype)

    # -------- 4. 对每个点做邻居多数投票 --------
    for i in range(n_cells):
        neigh_idx = indices[i]

        if not include_self:
            # 去掉自己（通常第一个就是自己，但为了稳妥再过滤一次）
            neigh_idx = neigh_idx[neigh_idx != i]

        # 如果样本很少，可能过滤自己后不足 n_neighbors 个，就直接用剩余的
        if len(neigh_idx) > n_neighbors:
            neigh_idx = neigh_idx[:n_neighbors]

        neigh_codes = label_codes[neigh_idx]

        # bincount 做多数投票，比 Python list.count 快很多
        counts = np.bincount(neigh_codes)
        majority_code = counts.argmax()
        new_codes[i] = majority_code

    new_labels = unique_labels[new_codes].astype(str)

    # -------- 5. 可选：写回 adata --------
    if new_key is not None:
        adata.obs[new_key] = new_labels

    return new_labels

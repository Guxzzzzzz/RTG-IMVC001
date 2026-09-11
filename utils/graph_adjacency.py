import torch
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def normalization_adj(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        The normalized adjacency matrix, the type is torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # add self-join
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # transform to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_similarity_matrix(features, method='heat'):
    """Get the similarity matrix"""
    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        # features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    return dist


def get_graph(features, topk=10, method='heat'):
    """Generate graph adjacency matrix using different similarity methods"""
    dist = get_similarity_matrix(features, method=method)
    # print(dist)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered


def get_adjacency(features, n, topk=10, self_join=True, method='heat'):
    """Get the standardized adjacency matrix, sparse and dense"""
    # features = features.cpu().numpy()# to cpu
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = get_graph(features, topk, method)
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if self_join:
        adj = adj + sp.eye(adj.shape[0])  # add self-join
    # raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def transfer_S(dist1, dist2, mask):
    """Transfer graph rows without ever selecting an unavailable endpoint.

    The legacy in-place implementation modified ``dist1`` before using it to
    build ``dist2``.  At high missing rates this made the result depend on
    view order and allowed zero-vector samples to become arbitrary nearest
    neighbours.  Each target row is now built independently from immutable
    similarities: an observed row uses its own view, while a missing row uses
    the donor view.  Candidate columns must be observed in that same source
    view.  ``-inf`` is intentional because a zero heat affinity can be a
    legitimate underflow value and must not tie with an invalid endpoint.
    """
    first = np.asarray(dist1)
    second = np.asarray(dist2)
    observed = np.asarray(mask, dtype=bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('similarity matrices must be equal-sized 2-D arrays')
    if first.shape[0] != first.shape[1]:
        raise ValueError('similarity matrices must be square')
    if observed.shape != (first.shape[0], 2):
        raise ValueError(
            'mask must have shape ({}, 2), got {}'.format(
                first.shape[0], observed.shape
            )
        )
    if np.any(observed.sum(axis=1) == 0):
        raise ValueError('each sample must retain at least one observed view')

    originals = (first, second)
    transferred = []
    for target_view in range(2):
        donor_view = 1 - target_view
        result = np.full_like(first, -np.inf)
        target_rows = observed[:, target_view]
        donor_rows = ~target_rows
        if np.any(target_rows):
            result[np.ix_(target_rows, observed[:, target_view])] = (
                originals[target_view][
                    np.ix_(target_rows, observed[:, target_view])
                ]
            )
        if np.any(donor_rows):
            result[np.ix_(donor_rows, observed[:, donor_view])] = (
                originals[donor_view][
                    np.ix_(donor_rows, observed[:, donor_view])
                ]
            )
        transferred.append(result)
    return transferred[0], transferred[1]


def get_edges(dist, topk=10):
    """Through the similarity matrix, the graph structure is established"""
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered


def graph2adj(edges_unordered, n, self_join=True):
    """Convert the established graph structure into the adjacency matrix required by GCN"""
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_join:  # add self-join
        adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def get_miss_adjacency(features1, features2, mask, n, topk=10):
    """
    Get the adjacency matrix of all data (including missing data), 
    note that the self-connection matrix is added here, because 
    the data is processed
    """
    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    mask = mask.cpu().numpy()

    dist1, dist2 = get_similarity_matrix(features1, 'heat'), get_similarity_matrix(features2, 'heat')
    dist1, dist2 = transfer_S(dist1, dist2, mask)
    edges1_unordered, edges2_unordered = get_edges(dist1, topk), get_edges(dist2, topk)
    adj1, raw_adj1 = graph2adj(edges1_unordered, n)
    adj2, raw_adj2 = graph2adj(edges2_unordered, n)
    return adj1, raw_adj1, adj2, raw_adj2


def _edges_to_neighbor_lists(edges, n):
    """Convert legacy directed edges to one neighbour array per sample.

    The original ICMVC routine calls ``argpartition`` for ``topk + 1``
    entries and removes the self index afterwards.  Replacing that operation
    with a seemingly equivalent diagonal-masking top-k changes the selected
    neighbours whenever similarities tie (a frequent case for sparse text
    views such as BBC).  Keeping the exact legacy edge set is therefore
    necessary for a controlled baseline comparison.
    """
    neighbors = [[] for _ in range(n)]
    for source, target in edges:
        neighbors[int(source)].append(int(target))
    return [np.asarray(row, dtype=np.int64) for row in neighbors]


def _legacy_neighbor_lists(similarity, topk):
    return _edges_to_neighbor_lists(get_edges(similarity, topk),
                                    similarity.shape[0])


def _neighbor_lists_to_edges(neighbors):
    return [
        [source, int(target)]
        for source, row in enumerate(neighbors)
        for target in row
    ]


def _jaccard_distance(first, second):
    first_set = set(int(value) for value in first)
    second_set = set(int(value) for value in second)
    union = first_set | second_set
    if not union:
        return 0.0
    return 1.0 - len(first_set & second_set) / float(len(union))


def _perturb_transferred_neighbors(base_neighbors, donor_neighbors,
                                   donor_similarity, missing_mask,
                                   donor_observed_mask, replacement_ratio,
                                   rng):
    """Generate one degree-preserving transferred-topology perturbation.

    Only rows belonging to samples missing in the target view are changed.
    Replacement neighbours are drawn first from the donor-view two-hop
    neighbourhood and then, if necessary, from all donor-observed samples.
    Each row keeps the same out-degree as the transferred baseline graph.
    """
    perturbed = [row.copy() for row in base_neighbors]
    n = len(base_neighbors)
    observed_pool = np.flatnonzero(donor_observed_mask)

    for sample_index in np.flatnonzero(missing_mask):
        current = base_neighbors[sample_index].copy()
        degree = current.size
        if degree == 0:
            continue
        replace_count = min(
            degree,
            max(1, int(round(float(replacement_ratio) * degree))),
        )
        positions = rng.choice(degree, size=replace_count, replace=False)

        direct = donor_neighbors[sample_index]
        second_hop_rows = [
            donor_neighbors[int(neighbor)] for neighbor in direct
            if donor_neighbors[int(neighbor)].size > 0
        ]
        if second_hop_rows:
            two_hop = np.concatenate(second_hop_rows)
        else:
            two_hop = np.empty(0, dtype=np.int64)
        local_pool = np.unique(np.concatenate((direct, two_hop)))
        excluded = set(int(value) for value in current)
        excluded.add(int(sample_index))
        local_pool = np.asarray(
            [value for value in local_pool if int(value) not in excluded],
            dtype=np.int64,
        )

        if local_pool.size < replace_count:
            global_pool = np.asarray(
                [value for value in observed_pool
                 if int(value) not in excluded and int(value) != sample_index],
                dtype=np.int64,
            )
            pool = np.unique(np.concatenate((local_pool, global_pool)))
        else:
            pool = local_pool

        if pool.size == 0:
            continue
        take = min(replace_count, pool.size)
        similarities = np.maximum(donor_similarity[sample_index, pool], 0.0)
        probabilities = None
        if np.isfinite(similarities).all() and similarities.sum() > 0:
            probabilities = similarities / similarities.sum()
        replacements = rng.choice(
            pool, size=take, replace=False, p=probabilities
        )
        current[positions[:take]] = replacements
        perturbed[sample_index] = current

    return perturbed


def get_rtg_adjacency(features1, features2, mask, n, topk=10,
                      candidate_count=4, replacement_ratio=0.3,
                      random_state=2023):
    """Build ICMVC transferred graphs and RTG topology candidates.

    ``candidate_count`` includes the unperturbed transferred topology.  The
    additional candidates use degree-preserving neighbour replacement and a
    donor-neighbourhood bootstrap.  The returned ``topology`` dictionary is
    consumed by :class:`models.RTGIMVC.RTGIMVC`.
    """
    if candidate_count < 1:
        raise ValueError('candidate_count must be at least one')
    if not 0.0 <= float(replacement_ratio) <= 1.0:
        raise ValueError('replacement_ratio must lie in [0, 1]')

    features1_np = features1.detach().cpu().numpy()
    features2_np = features2.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    if mask_np.shape != (n, 2):
        raise ValueError(
            'mask must have shape ({}, 2), got {}'.format(n, mask_np.shape)
        )
    if np.any(mask_np.sum(axis=1) == 0):
        raise ValueError('each sample must retain at least one observed view')

    original_distances = [
        get_similarity_matrix(features1_np, 'heat'),
        get_similarity_matrix(features2_np, 'heat'),
    ]
    transferred = transfer_S(
        original_distances[0].copy(),
        original_distances[1].copy(),
        mask_np,
    )
    # Preserve the *exact* ICMVC transferred graph as candidate zero.  This
    # includes NumPy's legacy tie handling in get_edges; sparse data may have
    # thousands of equal similarities, so a reimplemented top-k is not an
    # equivalent baseline.
    base_neighbors = [
        _legacy_neighbor_lists(transferred[0], topk),
        _legacy_neighbor_lists(transferred[1], topk),
    ]
    donor_neighbors = [
        _legacy_neighbor_lists(original_distances[0], topk),
        _legacy_neighbor_lists(original_distances[1], topk),
    ]

    base_outputs = []
    candidate_adjs = []
    cover_radii = []
    for view_index in range(2):
        base_edges = _neighbor_lists_to_edges(base_neighbors[view_index])
        base_adj, base_raw_adj = graph2adj(base_edges, n)
        base_outputs.append((base_adj, base_raw_adj))

        view_candidates = [base_adj]
        view_cover = np.zeros(n, dtype=np.float32)
        donor_index = 1 - view_index
        missing = ~mask_np[:, view_index]
        for candidate_index in range(1, int(candidate_count)):
            rng = np.random.RandomState(
                int(random_state) + 1009 * view_index + 9176 * candidate_index
            )
            candidate_neighbors = _perturb_transferred_neighbors(
                base_neighbors[view_index],
                donor_neighbors[donor_index],
                original_distances[donor_index],
                missing,
                mask_np[:, donor_index],
                replacement_ratio,
                rng,
            )
            for sample_index in np.flatnonzero(missing):
                view_cover[sample_index] = max(
                    view_cover[sample_index],
                    _jaccard_distance(
                        base_neighbors[view_index][sample_index],
                        candidate_neighbors[sample_index],
                    ),
                )
            candidate_edges = _neighbor_lists_to_edges(candidate_neighbors)
            candidate_adj, _ = graph2adj(candidate_edges, n)
            view_candidates.append(candidate_adj)

        candidate_adjs.append(view_candidates)
        cover_radii.append(torch.from_numpy(view_cover))

    topology = {
        'candidate_adjs': candidate_adjs,
        'cover_radius': cover_radii,
        'missing_mask': [
            torch.from_numpy((~mask_np[:, 0]).copy()),
            torch.from_numpy((~mask_np[:, 1]).copy()),
        ],
        'candidate_count': int(candidate_count),
        'replacement_ratio': float(replacement_ratio),
        'certificate_kind': 'empirical_finite_cover',
    }
    return (
        base_outputs[0][0], base_outputs[0][1],
        base_outputs[1][0], base_outputs[1][1], topology,
    )

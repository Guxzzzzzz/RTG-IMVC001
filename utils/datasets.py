import math
import os
import re

import numpy as np
import scipy.io as sio
from scipy import sparse


def _repo_root():
    """Return the project root regardless of how train.py is launched."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _dense(value):
    """Convert scipy sparse or dense MATLAB data to a NumPy array."""
    if sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)


def _row_l2_normalize(value):
    """Normalize each sample while preserving zero rows safely."""
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norms, 1.0e-12)


def _subsample_aligned(X_list, y, max_samples=None, seed=2023, sampling="random"):
    """Select one deterministic aligned subset from every selected view."""
    if max_samples is None or y.shape[0] <= max_samples:
        return X_list, y
    rng = np.random.default_rng(seed)
    if sampling == "stratified":
        classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        expected = counts.astype(np.float64) * max_samples / y.shape[0]
        take = np.floor(expected).astype(np.int64)
        remainder = int(max_samples - take.sum())
        # Fractional apportionment preserves the original class distribution;
        # a seeded tie-break makes the chosen subset exactly reproducible.
        tie_break = rng.random(classes.shape[0])
        order = np.lexsort((tie_break, -(expected - take)))
        for class_id in order:
            if remainder == 0:
                break
            if take[class_id] < counts[class_id]:
                take[class_id] += 1
                remainder -= 1
        selected = []
        for class_id, class_take in enumerate(take):
            class_indices = np.flatnonzero(inverse == class_id)
            selected.append(
                rng.choice(class_indices, size=int(class_take), replace=False)
            )
        indices = np.sort(np.concatenate(selected))
    elif sampling == "random":
        indices = np.sort(rng.choice(y.shape[0], size=max_samples, replace=False))
    else:
        raise ValueError(
            f"Unsupported sampling policy {sampling!r}; "
            "choose 'random' or 'stratified'"
        )
    return [x[indices] for x in X_list], y[indices]


def _sample_matrix(value, n_samples, name):
    """Convert one dense/sparse MATLAB view to [samples, flat_features]."""
    value = _dense(value)
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got {value.shape}")
    if value.shape[0] == n_samples:
        result = value.reshape(n_samples, -1)
    elif value.ndim == 2 and value.shape[1] == n_samples:
        result = value.T
    else:
        raise ValueError(
            f"{name} has no sample axis of length {n_samples}; got {value.shape}"
        )
    return result.astype(np.float32, copy=False)


def _ordered_mat_views(mat, name, view_keys=None):
    """Return all source views and names from common MATLAB conventions."""
    if view_keys is not None:
        missing = [key for key in view_keys if key not in mat]
        if missing:
            raise KeyError(
                f"{name}.mat is missing configured view fields {missing}; "
                f'available keys: {sorted(k for k in mat if not k.startswith("__"))}'
            )
        return [mat[key] for key in view_keys], list(view_keys)

    if "X" in mat and np.asarray(mat["X"]).dtype == object:
        cells = np.asarray(mat["X"]).ravel(order="C")
        return list(cells), [f"X[{index}]" for index in range(cells.size)]

    numbered = [key for key in mat if re.fullmatch(r"X\d+", key)]
    numbered.sort(key=lambda key: int(key[1:]))
    if numbered:
        return [mat[key] for key in numbered], numbered

    raise KeyError(
        f"{name}.mat must contain an X cell or numbered X1..XV fields; "
        f'available keys: {sorted(k for k in mat if not k.startswith("__"))}'
    )


def _load_two_view_mat(
    mat,
    view_indices=(0, 1),
    name="dataset",
    normalize_views=False,
    max_samples=None,
    sample_seed=2023,
    sampling="random",
    view_keys=None,
    label_key=None,
    expected_dims=None,
):
    """Load common MATLAB multi-view layouts under a disclosed 2-view protocol."""
    if len(view_indices) != 2 or min(view_indices) < 0:
        raise ValueError(
            f"{name} requires exactly two non-negative view indices; "
            f"got {view_indices}"
        )

    if label_key is None:
        label_key = next(
            (key for key in ("Y", "gt", "truth", "label", "labels") if key in mat),
            None,
        )
    if label_key is None or label_key not in mat:
        raise KeyError(
            f"{name}.mat does not contain the configured label field; "
            f'available keys: {sorted(k for k in mat if not k.startswith("__"))}'
        )
    raw_y = np.asarray(mat[label_key]).reshape(-1)
    if raw_y.size == 0 or not np.isfinite(raw_y).all():
        raise ValueError(f"{name}.mat labels are empty or contain NaN/Inf")
    _, y = np.unique(raw_y, return_inverse=True)
    y = y.astype(np.int64, copy=False)

    source_views, source_names = _ordered_mat_views(mat, name=name, view_keys=view_keys)
    if max(view_indices) >= len(source_views):
        raise ValueError(
            f"{name}.mat has {len(source_views)} views, but requested {view_indices}"
        )
    X_list = [
        _sample_matrix(source_views[index], y.shape[0], source_names[index])
        for index in view_indices
    ]

    if expected_dims is not None:
        actual_dims = tuple(view.shape[1] for view in X_list)
        if actual_dims != tuple(expected_dims):
            raise ValueError(
                f"{name}.mat selected feature dimensions {actual_dims} do not "
                f"match configured encoder inputs {tuple(expected_dims)}"
            )

    if normalize_views:
        X_list = [_row_l2_normalize(x) for x in X_list]

    if any(x.shape[0] != y.shape[0] for x in X_list):
        raise ValueError(
            f"{name}.mat sample counts do not match: "
            f"views={[x.shape for x in X_list]}, y={y.shape}"
        )
    if not np.isfinite(y).all() or any(not np.isfinite(x).all() for x in X_list):
        raise ValueError(f"{name}.mat contains NaN or Inf values")

    X_list, y = _subsample_aligned(
        X_list,
        y,
        max_samples=max_samples,
        seed=sample_seed,
        sampling=sampling,
    )
    return X_list, [y, y.copy()]


def load_data(config, train_dir=False):
    data_name = config["dataset"]
    X_list = []
    Y_list = []
    main_dir = _repo_root()
    if data_name in ["handwritten"]:
        mat = sio.loadmat(os.path.join(main_dir, "data", data_name + ".mat"))
        X = mat["X"][0]
        y = np.squeeze(mat["Y"]).astype(np.int64)
        view_indices = tuple(config.get("view_indices", (0, 1)))
        if max(view_indices) >= len(X):
            raise ValueError(
                f"handwritten.mat has {len(X)} views, but requested {view_indices}"
            )
        for view in view_indices:
            x = _dense(X[view]).astype(np.float32)
            if config.get("normalize_views", True):
                x = _row_l2_normalize(x)
            X_list.append(x)
            Y_list.append(y.copy())
    elif data_name in ["MSRC_v1"]:
        mat = sio.loadmat(os.path.join(main_dir, "data", data_name + ".mat"))
        if "msr2" in mat and "msr3" in mat:
            for view in ["msr2", "msr3"]:
                X_list.append(_dense(mat[view]).astype("float32"))
                Y_list.append(np.squeeze(mat["truth"]))
        elif "X1" in mat and "gt" in mat:
            X_list.append(_dense(mat["X1"]).astype("float32"))
            X_list.append(_dense(mat["X2"]).astype("float32"))
            Y_list.append(np.squeeze(mat["gt"]))
            Y_list.append(np.squeeze(mat["gt"]))
        else:
            raise KeyError(
                f"MSRC_v1 fields do not match; available fields: {list(mat.keys())}"
            )
    elif data_name in ["NoisyMNIST"]:
        mat = sio.loadmat(os.path.join(main_dir, "data", data_name + ".mat"))
        if "X1" in mat and "gt" in mat:
            X_list.append(_dense(mat["X1"]).astype("float32"))
            X_list.append(_dense(mat["X2"]).astype("float32"))
            Y_list.append(np.squeeze(mat["gt"]))
            Y_list.append(np.squeeze(mat["gt"]))
        else:
            tune = DataSet_NoisyMNIST(mat["XV1"], mat["XV2"], mat["tuneLabel"])
            test = DataSet_NoisyMNIST(mat["XTe1"], mat["XTe2"], mat["testLabel"])
            X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
            X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
            Y_list.append(
                np.concatenate(
                    [np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]
                )
            )
            Y_list.append(
                np.concatenate(
                    [np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]
                )
            )
    elif data_name in ["BBC", "NGs", "3V_Fashion_MV", "Caltech-5V", "3-sources"]:
        source_file = config.get(
            "source_file", os.path.join(main_dir, "data", data_name + ".mat")
        )
        mat = sio.loadmat(source_file)
        X_list, Y_list = _load_two_view_mat(
            mat,
            view_indices=tuple(config.get("view_indices", (0, 1))),
            name=data_name,
            normalize_views=bool(config.get("normalize_views", False)),
            max_samples=config.get("max_samples"),
            sample_seed=int(config.get("sample_seed", 2023)),
            sampling=config.get("sampling", "random"),
            view_keys=config.get("view_keys"),
            label_key=config.get("label_key"),
            expected_dims=(
                config["Autoencoder"]["gcnEncoder1"][0],
                config["Autoencoder"]["gcnEncoder2"][0],
            ),
        )
    else:
        raise Exception("Undefined data_name")
    if len(X_list) != 2 or len(Y_list) != 2:
        raise ValueError(
            f"{data_name} must resolve to exactly two aligned views; got {len(X_list)} feature and {len(Y_list)} label arrays"
        )
    expected_dims = (
        int(config["Autoencoder"]["gcnEncoder1"][0]),
        int(config["Autoencoder"]["gcnEncoder2"][0]),
    )
    actual_dims = tuple((int(view.shape[1]) for view in X_list))
    if actual_dims != expected_dims:
        raise ValueError(
            f"{data_name} loaded dimensions {actual_dims}, but its encoders expect {expected_dims}"
        )
    if not np.array_equal(Y_list[0], Y_list[1]):
        raise ValueError(f"{data_name} returned inconsistent labels across views")
    if np.unique(Y_list[0]).size != int(config["n_clusters"]):
        raise ValueError(
            f"{data_name} has {np.unique(Y_list[0]).size} classes, but n_clusters={config['n_clusters']}"
        )
    source_view_count = config.get("source_view_count")
    if source_view_count is not None:
        selected = tuple((index + 1 for index in config.get("view_indices", (0, 1))))
        selected_names = config.get("source_view_names")
        name_suffix = f" ({', '.join(selected_names)})" if selected_names else ""
        print(
            f"[datasets.py] Two-view protocol: selected source views {selected} of {source_view_count}{name_suffix}"
        )
    print(
        f"[datasets.py] Loaded {data_name}: X shapes = {[x.shape for x in X_list]}, Y shape = {Y_list[0].shape}"
    )
    return (X_list, Y_list)


def _balanced_batch_bounds(total, batch_size):
    """Partition every sample into non-empty batches no larger than the cap."""
    total = int(total)
    batch_size = int(batch_size)
    if total <= 0 or batch_size <= 0:
        raise ValueError("total and batch_size must be positive")
    batch_count = int(math.ceil(float(total) / batch_size))
    base_size, remainder = divmod(total, batch_count)
    start = 0
    for batch_index in range(batch_count):
        size = base_size + int(batch_index < remainder)
        end = start + size
        yield start, end
        start = end


def next_batch(X1, X2, batch_size):
    """Generate balanced paired batches while retaining every sample."""
    if X1.shape[0] != X2.shape[0]:
        raise ValueError("paired views must contain the same number of samples")
    for batch_index, (start_idx, end_idx) in enumerate(
        _balanced_batch_bounds(X1.shape[0], batch_size)
    ):
        yield (
            X1[start_idx:end_idx, ...],
            X2[start_idx:end_idx, ...],
            batch_index + 1,
        )


def next_batch_gt(X1, X2, gt, batch_size):
    """Generate paired batches with labels."""
    if X1.shape[0] != X2.shape[0] or X1.shape[0] != gt.shape[0]:
        raise ValueError("paired views and labels must have equal length")
    for batch_index, (start_idx, end_idx) in enumerate(
        _balanced_batch_bounds(X1.shape[0], batch_size)
    ):
        yield (
            X1[start_idx:end_idx, ...],
            X2[start_idx:end_idx, ...],
            gt[start_idx:end_idx, ...],
            batch_index + 1,
        )


def next_batch_list(X, batch_size):
    """Generate batches for a list of aligned views."""
    if not X or any(view.shape[0] != X[0].shape[0] for view in X):
        raise ValueError("all views must be non-empty and aligned")
    total = X[0].shape[0]
    for batch_index, (start_idx, end_idx) in enumerate(
        _balanced_batch_bounds(total, batch_size)
    ):
        yield [view[start_idx:end_idx, ...] for view in X], batch_index + 1


class DataSet_NoisyMNIST(object):

    def __init__(
        self, images1, images2, labels, fake_data=False, one_hot=False, dtype=np.float32
    ):
        """Construct a NoisyMNIST data set."""
        t = 2
        if dtype not in (np.uint8, np.float32):
            raise TypeError(f"Invalid image dtype {dtype!r}, expected uint8 or float32")

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert (
                images1.shape[0] == labels.shape[0]
            ), f"images1.shape: {images1.shape} labels.shape: {labels.shape}"
            assert (
                images2.shape[0] == labels.shape[0]
            ), f"images2.shape: {images2.shape} labels.shape: {labels.shape}"
            self._num_examples = images1.shape[0] // t

            if dtype == np.float32 and images1.dtype != np.float32:
                images1 = images1.astype(np.float32)
            if dtype == np.float32 and images2.dtype != np.float32:
                images2 = images2.astype(np.float32)

        self._images1 = images1[::t]
        self._images2 = images2[::t]
        self._labels = labels[::t]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next batch."""
        if fake_data:
            fake_image = [1] * 784
            fake_label = [1] + [0] * 9 if self.one_hot else 0
            return (
                [fake_image for _ in range(batch_size)],
                [fake_image for _ in range(batch_size)],
                [fake_label for _ in range(batch_size)],
            )

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return (
            self._images1[start:end],
            self._images2[start:end],
            self._labels[start:end],
        )

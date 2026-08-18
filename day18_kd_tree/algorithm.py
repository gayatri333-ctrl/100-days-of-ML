import numpy as np


class KDNode:
    """A single node in the KD-Tree."""
    def __init__(self, point, index, left=None, right=None, axis=0):
        self.point = point      # the actual data point (coordinates)
        self.index = index      # index into the original training array (to look up its label)
        self.left = left        # subtree containing points "less than" this node along `axis`
        self.right = right      # subtree containing points "greater than" this node along `axis`
        self.axis = axis        # which dimension/feature this node splits on


def build_kd_tree(points, indices, depth=0):
    """
    Recursively builds a KD-Tree.

    The core idea: at each level of the tree, pick a dimension to split
    on (cycling through dimensions as depth increases: x, then y, then x,
    then y, ...). Sort points along that dimension, put the median point
    at this node, and recursively build left/right subtrees from the
    points below/above the median.

    This creates a structure where "nearby in space" tends to mean
    "nearby in the tree" too — which is what lets us skip large chunks
    of the data during search instead of checking every point.
    """
    if len(points) == 0:
        return None

    n_features = points.shape[1]
    axis = depth % n_features  # cycle through dimensions as we go deeper

    # Sort points (and their original indices) along the chosen axis
    sorted_order = np.argsort(points[:, axis])
    points_sorted = points[sorted_order]
    indices_sorted = indices[sorted_order]

    median_idx = len(points_sorted) // 2

    node = KDNode(
        point=points_sorted[median_idx],
        index=indices_sorted[median_idx],
        axis=axis,
    )

    # Recursively build left (smaller values) and right (larger values) subtrees
    node.left = build_kd_tree(points_sorted[:median_idx], indices_sorted[:median_idx], depth + 1)
    node.right = build_kd_tree(points_sorted[median_idx + 1:], indices_sorted[median_idx + 1:], depth + 1)

    return node


def _squared_distance(a, b):
    """Squared Euclidean distance (skip the sqrt — we only need it for comparisons)."""
    return np.sum((a - b) ** 2)


def _search_knn(node, target, k, heap):
    """
    Recursively searches the tree for the k nearest points to `target`.

    `heap` is a simple list of (distance, index, point) tuples we keep
    sorted, holding the best k candidates found so far.

    The key optimization: after exploring the "near" branch, we check
    whether the "far" branch could POSSIBLY contain a closer point than
    what we've already found. If not, we skip it entirely — this is
    what makes KD-Tree search faster than brute force.
    """
    if node is None:
        return

    dist = _squared_distance(target, node.point)

    # Add this node to our candidate list, keep only the k best
    heap.append((dist, node.index, node.point))
    heap.sort(key=lambda x: x[0])
    if len(heap) > k:
        heap.pop()

    axis = node.axis
    diff = target[axis] - node.point[axis]

    # Decide which side to search first (the side the target point is "closer" to)
    near_branch = node.left if diff < 0 else node.right
    far_branch = node.right if diff < 0 else node.left

    _search_knn(near_branch, target, k, heap)

    # Only search the far branch if it could possibly contain a closer point
    # than our current worst candidate — this is the pruning step that
    # makes KD-Tree faster than brute-force search.
    if len(heap) < k or diff ** 2 < heap[-1][0]:
        _search_knn(far_branch, target, k, heap)


class KDTree:
    """
    Wrapper class providing a clean fit/query interface around the
    recursive KD-Tree build and search functions above.
    """

    def __init__(self):
        self.root = None

    def fit(self, X):
        X = np.array(X)
        indices = np.arange(X.shape[0])
        self.root = build_kd_tree(X, indices, depth=0)
        return self

    def query(self, target, k=1):
        """
        Returns (distances, indices) of the k nearest neighbors to `target`,
        sorted from closest to farthest.
        """
        heap = []
        _search_knn(self.root, np.array(target), k, heap)
        heap.sort(key=lambda x: x[0])

        distances = np.sqrt([h[0] for h in heap])  # convert back from squared distance
        indices = [h[1] for h in heap]
        return distances, indices


if __name__ == "__main__":
    # Quick manual test + a speed comparison against brute-force search
    import time

    np.random.seed(42)
    X = np.random.rand(2000, 2) * 100  # 2000 random 2D points
    query_point = np.array([50.0, 50.0])
    k = 5

    # ---- KD-Tree search ----
    tree = KDTree()
    tree.fit(X)

    start = time.time()
    distances, indices = tree.query(query_point, k=k)
    kd_time = time.time() - start

    print(f"KD-Tree nearest {k} neighbors (indices): {indices}")
    print(f"KD-Tree distances: {np.round(distances, 4)}")
    print(f"KD-Tree search time: {kd_time*1000:.4f} ms")

    # ---- Brute-force search (compute distance to every point) ----
    start = time.time()
    all_distances = np.sqrt(np.sum((X - query_point) ** 2, axis=1))
    brute_indices = np.argsort(all_distances)[:k]
    brute_distances = all_distances[brute_indices]
    brute_time = time.time() - start

    print(f"\nBrute-force nearest {k} neighbors (indices): {list(brute_indices)}")
    print(f"Brute-force distances: {np.round(brute_distances, 4)}")
    print(f"Brute-force search time: {brute_time*1000:.4f} ms")

    print(f"\nResults match: {sorted(indices) == sorted(brute_indices.tolist())}")
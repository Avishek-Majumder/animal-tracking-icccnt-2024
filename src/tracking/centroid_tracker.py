"""
Centroid-based multi-object tracker.

We follow a simple, interpretable approach:

- Each detected object in a frame is represented by its centroid (x, y).
- For each new frame, we match current centroids to existing tracked objects
  using Euclidean distance.
- If the closest distance is below a threshold, we update that object's
  centroid and reset its "disappeared" counter.
- If an object cannot be matched for several consecutive frames,
  we consider it disappeared and remove it.
- Any unmatched centroid in the current frame is treated as a new object.

This is in line with the tracking approach we described in the paper:
a lightweight, centroid-based tracker suitable for lamb tracking
in a relatively constrained environment.
"""

import numpy as np


class CentroidTracker:
    """
    A simple centroid-based multi-object tracker.

    Attributes
    ----------
    next_object_id : int
        The ID that will be assigned to the next new object.
    objects : dict[int, np.ndarray]
        Mapping from object_id to its centroid (x, y).
    disappeared : dict[int, int]
        Mapping from object_id to the number of consecutive frames
        in which it has not been observed.
    max_distance : float
        Maximum allowed distance (in pixels) between an existing
        object centroid and a new detection centroid for them
        to be considered a match.
    max_disappeared : int
        Maximum number of consecutive frames an object is allowed
        to be missing before we delete it.
    """

    def __init__(self, max_distance=50.0, max_disappeared=10):
        """
        Parameters
        ----------
        max_distance : float, optional
            Maximum distance in pixels for matching detections to
            existing tracked objects. If the distance is larger than
            this threshold, the detection is treated as a new object.
        max_disappeared : int, optional
            Number of consecutive frames an object can be missing
            before being deregistered.
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}

        self.max_distance = float(max_distance)
        self.max_disappeared = int(max_disappeared)

    def _register(self, centroid):
        """
        Register a new object with the given centroid.
        """
        object_id = self.next_object_id
        self.objects[object_id] = np.array(centroid, dtype=float)
        self.disappeared[object_id] = 0
        self.next_object_id += 1

        return object_id

    def _deregister(self, object_id):
        """
        Remove an object from our internal state.
        """
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, input_centroids):
        """
        Update the tracker with the centroids detected in the current frame.

        Parameters
        ----------
        input_centroids : list[tuple[float, float]] or np.ndarray
            List of (x, y) centroids for the current frame.

        Returns
        -------
        dict[int, np.ndarray]
            The updated mapping from object_id to centroid.
        """
        # Ensure we have a NumPy array representation
        if len(input_centroids) == 0:
            # No detections in this frame:
            # increase disappeared counter for each existing object
            # and deregister if necessary.
            to_deregister = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    to_deregister.append(object_id)

            for object_id in to_deregister:
                self._deregister(object_id)

            return self.objects

        input_centroids = np.array(input_centroids, dtype=float)

        # If we are not tracking anything yet, register all input centroids.
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self._register(centroid)
            return self.objects

        # Otherwise, we try to match existing objects to new centroids.
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()), dtype=float)

        # Compute pairwise distance matrix between existing objects and new detections.
        # distances[i, j] = distance between object i and detection j.
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        # For matching, we:
        # 1) sort object rows by their minimum distance to any detection
        # 2) in that order, assign each object to its closest detection
        #    if it is within the max_distance and not already taken.
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            distance = distances[row, col]
            if distance > self.max_distance:
                # Too far away: do not associate, treat as potentially new object later.
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle objects that were not matched with any detection.
        unused_rows = set(range(distances.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        # Any detection that was not assigned to an existing object becomes a new object.
        unused_cols = set(range(distances.shape[1])).difference(used_cols)
        for col in unused_cols:
            centroid = input_centroids[col]
            self._register(centroid)

        return self.objects

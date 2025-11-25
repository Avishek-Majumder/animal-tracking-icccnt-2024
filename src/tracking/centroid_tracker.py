# src/tracking/centroid_tracker.py

import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=50, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}          # object_id -> centroid (x, y)
        self.disappeared = {}      # object_id -> frames since last seen
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # input_centroids: list of (x, y) for current frame
        if len(input_centroids) == 0:
            # mark existing objects disappeared
            ids = list(self.disappeared.keys())
            for object_id in ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        # match existing object centroids to new centroids using distance
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # compute distance matrix
        distances = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2
        )

        # rows: existing objects, cols: new detections
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # any unmatched objects → disappeared++
        unused_rows = set(range(distances.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # any unmatched input centroids → new objects
        unused_cols = set(range(distances.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects

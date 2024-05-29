import numpy
import numpy as np
from utils.io import BaseTrackResultReader


def cosine_distance(f1, f2):
    return 1 - np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def track_distance(source_f, dest_f):
    filter1 = np.array([i[0:2048].all == 0 for i in source_f])
    filter2 = np.array([i[2048:4096].all == 0 for i in source_f])
    filter3 = np.array([i[4096:6044].all == 0 for i in source_f])
    filter4 = np.array([i[0:2048].all == 0 for i in dest_f])
    filter5 = np.array([i[2048:4096].all == 0 for i in dest_f])
    filter6 = np.array([i[4096:6044].all == 0 for i in dest_f])
    sum_distance = 0
    sf_candidate = source_f[np.logical_not(filter1 | filter2 | filter3)]
    df_candidate = dest_f[np.logical_not(filter4 | filter5 | filter6)]
    for i in sf_candidate:
        sum_distance += min([cosine_distance(i, j) for j in df_candidate])
    return sum_distance / len(sf_candidate)


class UnreasonMatch(BaseTrackResultReader):

    def __init__(self, video_path, track_result_path):
        super().__init__(video_path, track_result_path)
        self.unreason = {}
    def worker(self, img, results):
        results = np.array(results, dtype=np.float32)
        track_ids = results[:, 1]
        for id_ in track_ids:
            u = track_ids.tolist()
            u: list
            u.remove(id_)
            if id_ not in self.unreason:
                self.unreason[id_] = set()
            for u_ in u:
                self.unreason[id_].add(u_)


if __name__ == "__main__":
    # from dataLoader import Datasets
    #
    # datasets = Datasets(
    #     "/home/zlj/Excavator_ReID/Trained_ReID_features/1716362426.0804555_bank.npy")
    # bank_x = datasets.data
    # bank_y = datasets.target
    #
    # bank_d = bank_x[bank_y == 2.]
    # bank_s = bank_x[bank_y == 4.]
    # print(track_distance(bank_s, bank_d))
    with UnreasonMatch('/home/zlj/Excavator_ReID/121241-123840.mp4',
                                '/home/zlj/Excavator_ReID/results.txt') as um:
        um.rus()

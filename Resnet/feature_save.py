import os
import time

import numpy
import numpy as np


class Feature_map:
    def __init__(self):
        self.data = np.empty(shape=(0, 6146))

    def _add_feature(self, sig_result):
        self.data = np.concatenate((self.data, sig_result), axis=0)

    @property
    def result(self):
        return self.data

    def massage_build(self, frame_id, track_id, features):
        assert len(track_id) == len(frame_id) == len(features)
        s_massage = numpy.zeros(shape=(len(track_id), features.shape[1] + 2))
        s_massage[:, 0], s_massage[:, 1], s_massage[:, 2:] = numpy.array(frame_id), numpy.array(track_id), features

        return s_massage

    def __call__(self, frame_id, track_id, features):
        s_massage = self.massage_build(frame_id, track_id, features)
        self._add_feature(s_massage)


class Feature_bank(Feature_map):
    def __init__(self):
        super(Feature_bank, self).__init__()
        self.bank = {}
        self.temper = {}

    def _cosine_distance(self, a, b):
        return 1 - numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

    def _distance_gate(self, features):
        distance = 0
        if len(features) < 2:
            return 0

        for i in range(len(features) - 1):
            for j in features[i + 1:]:
                distance += self._cosine_distance(j, features[i])
        return distance * 2 / (len(features) * (len(features) - 1))

    def _feature_check(self, track_id, target_feature):
        if self.temper[track_id] == 0:
            # print(f'{track_id} freezed!')
            # 温度降到冰点，重置
            return True

        if len(self.bank[track_id]) < 30:
            gate_range = self.bank[track_id]
            distance_gate = self._distance_gate(gate_range)
            length = len(self.bank[track_id])
            distance = distance_gate * length * (length - 1) / 2
        else:
            gate_range = self.bank[track_id][-30:]
            distance_gate = self._distance_gate(gate_range)
            length = 30
            distance = distance_gate * length * (length - 1) / 2

        for i in gate_range:
            distance += self._cosine_distance(i, target_feature)

        distance_new = distance * 2 / ((length + 1) * length)

        if distance_new > distance_gate:
            return True
        else:
            return False

    def bank_update(self, massage):
        self.temper[massage[1]] = 40
        x = massage[2:].reshape(1, -1)
        self.bank[massage[1]] = numpy.concatenate((self.bank[massage[1]], x), axis=0)

    def _add_feature(self, massages):
        for massage in massages:
            if massage[1] not in self.bank:
                self.bank[massage[1]] = numpy.empty(shape=(0, 6144))
                self.bank_update(massage)
            else:
                if self._feature_check(track_id=massage[1], target_feature=massage[2:]):
                    self.bank_update(massage)
                else:
                    self.temper[massage[1]] -= 1

    @property
    def result(self):
        res = numpy.empty(shape=(0, 6145))
        for k, v in self.bank.items():
            for vv in v:
                x = numpy.concatenate(([k], vv))
                x = numpy.reshape(x, newshape=(1, -1))
                res = numpy.concatenate([res, x], axis=0)
        return res


if __name__ == "__main__":
    feature_map = Feature_bank()
    feature_map([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], numpy.random.rand(6, 6144))
    numpy.save("./test", feature_map.result)
    print(numpy.load("./test.npy"))

import os

import numpy


class Feature_bank:
    def __init__(self):
        self.bank = {}
        self.timestamp = {}
        self.temper = {}
        self.bank_path = os.path.join(os.getcwd(), "feature_bank")
        os.makedirs(self.bank_path, exist_ok=True)
        self.bank_data = {}

    def _cosine_distance(self, a, b):
        return 1 - numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

    def _distance_gate(self, features):
        distance = 0
        if len(features) < 2:
            return 0

        for i in range(len(features) - 1):
            for j in features[i+1:]:
                distance += self._cosine_distance(j, features[i])
        return distance * 2 / (len(features) * (len(features) - 1))

    def _add_feature(self, track_id, frame_id, target_feature):
        self.bank[track_id].append(target_feature.tolist())
        self.timestamp[track_id].append(frame_id)
        # print(len(self.timestamp[track_id]))
        self.temper[track_id] = 240

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

    def _target_feature_build(self, features):
        target_feature = numpy.zeros((2048 * 3))
        for i in range(3):
            target_feature[i * 2048: i * 2048 + 2048] = features[i]
        return target_feature

    def __call__(self, track_id, frame_id, features):
        target_feature = self._target_feature_build(features)

        if track_id not in self.bank:
            self.bank[track_id] = []
            self.timestamp[track_id] = []
            self._add_feature(track_id, frame_id, target_feature)

        elif len(self.bank[track_id]) < 2:
            self._add_feature(track_id, frame_id, target_feature)

        else:
            if self._feature_check(track_id, target_feature):
                self._add_feature(track_id, frame_id, target_feature)
            else:
                # 加入失败，降温
                # print(f'{track_id} temper lose 1')
                self.temper[track_id] -= 1

    @property
    def result(self):
        return self.bank, self.timestamp


class Feature_map(Feature_bank):
    def __init__(self):
        self.bank = {}
        self.timestamp = {}
        self.temper = {}
        self.map_path = os.path.join(os.getcwd(), "feature_map")
        os.makedirs(self.map_path, exist_ok=True)
        self.bank_data = {}

    def _add_feature(self, track_id, frame_id, target_feature):
        self.bank_data[track_id] = open(self.bank[track_id], 'ab')
        self.bank_data[track_id].write(target_feature.tobytes())
        self.bank_data[track_id].close()
        self.timestamp[track_id].append(frame_id)

    def __call__(self, track_id, frame_id, features):
        target_feature = self._target_feature_build(features)

        if track_id not in self.bank:
            self.bank[track_id] = os.path.join(self.map_path, track_id)
            self.timestamp[track_id] = []
            self._add_feature(track_id, frame_id, target_feature)

        else:
            self._add_feature(track_id, frame_id, target_feature)



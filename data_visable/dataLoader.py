import numpy 
import json
from tqdm import tqdm
 
class Datasets:
    
    def __init__(self, path) -> None:
        self.length = 0
        self.data = numpy.array([])
        self.target = []
        self.data_load(path)
        
    def data_load(self, path):
        with open(path, 'r') as file:
            data = json.loads(file.read())
        
        classes = data.keys()
        for classe in classes:
            for feature in data[classe]:
                self.data = numpy.append(self.data, feature)
                self.target.append(classe)
                self.length += 1
        self.data = self.data.reshape(self.length, -1)
        self.target = numpy.array(self.target)
            

class FullDatasets(Datasets):
    def __init__(self, path) :
        super().__init__(path)
        self.have_loaded = False

    def data_load(self, path):
        with open(path, 'r') as file:
            data_map = json.loads(file.read())

        for label in data_map:
            with open(data_map[label], 'rb') as f:
                tex = f.read()
                length = int(len(tex) / 49152)
                features = numpy.frombuffer(tex, dtype=numpy.float64) 
                self.data = numpy.append(self.data, features)
                self.length += length
                for i in range(length):
                    self.target.append(label)
        self.data = self.data.reshape(self.length, -1)
        self.target = numpy.array(self.target)
        self.have_loaded = True
    
    # def 
    


if __name__ == "__main__":
    path = "/home/zlj/Excavator_ReID/feature_map.json"
    data = FullDatasets(path)
    
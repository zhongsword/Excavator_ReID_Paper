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
        temp_data = numpy.load(path)
        self.classes = set(temp_data[:, 0])
        self.target = temp_data[:, 0]
        self.data = temp_data[:, 1:]
            

class FullDatasets(Datasets):

    def data_load(self, path):
        temp_data = numpy.load(path)
        self.classes = set(temp_data[:, 1])
        self.target = temp_data[:, 1]
        self.data = temp_data[:, 2:] 
    

if __name__ == "__main__":
    path = "/Trained_ReID_features/1716362426.0804555_bank.npy"
    data = Datasets(path)
    
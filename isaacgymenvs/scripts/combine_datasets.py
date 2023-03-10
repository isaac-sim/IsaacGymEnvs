""" Script to combine two datasets into a single dataset 

Assume all dataset files are located in same directory """

import argparse 
import pathlib
import yaml

from typing import List 
from dataclasses import dataclass, field


@dataclass
class MotionData:
    file: str 
    weight: float 

    @staticmethod
    def from_dict(dict):
        return MotionData(file=dict['file'], weight=dict['weight'])
    
    def to_dict(self):
        return {'file': self.file, 'weight': self.weight}

@dataclass
class MotionDataset:
    data: List[MotionData] = field(default_factory=list)

    @staticmethod 
    def from_dict(dict):
        return MotionDataset([MotionData.from_dict(d) for d in dict['motions']])

    def to_dict(self):
        return {'motions': [data.to_dict() for data in self.data]}
    
    def __len__(self):
        return len(self.data)
    
    def get_total_weight(self):
        return sum([d.weight for d in self.data])
    
    def normalize_weight(self):
        total_weight = self.get_total_weight()
        for d in self.data:
            d.weight /= total_weight

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-filepaths', nargs='+', type=str)
    parser.add_argument('-o', '--output-filepath', type=str)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()

    datasets = []
    for fp in args.input_filepaths: 
        fp = pathlib.Path(fp)
        with open(fp, 'r') as file:
            dataset = yaml.load(file, yaml.SafeLoader)
        datasets.append(MotionDataset.from_dict(dataset))
    
    # Combine the datasets
    combined_dataset = MotionDataset([])
    for dataset in datasets:
        combined_dataset.data.extend(dataset.data)
    combined_dataset.normalize_weight()
    
    with open(args.output_filepath, 'w') as file:
        yaml.dump(combined_dataset.to_dict(), file)
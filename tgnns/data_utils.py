import os
import random
import pickle
import torch
from tqdm import tqdm
from typing_extensions import Literal
from .graph_utils import GraphUtils

class DataUtils:
    dataset_path=os.path.join('..','data','tgnns')
    @staticmethod
    def save_to_pickle(data,file_name:str,dir_type:Literal['graph','train','val','test'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")
    
    @staticmethod
    def load_from_pickle(file_name:str,dir_type:Literal['graph','train','val','test'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

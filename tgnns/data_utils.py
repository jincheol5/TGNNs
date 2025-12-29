import os
import random
import pickle
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal
from .graph_utils import GraphUtils

class DataUtils:
    dataset_path=os.path.join('..','data','tgnns')
    @staticmethod
    def save_to_pickle(data,file_name:str,dir_type:Literal['graph','dataset'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")
    
    @staticmethod
    def load_from_pickle(file_name:str,dir_type:Literal['graph','dataset'],num_nodes:Literal[20,50,100,500,1000]=20):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='test':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,f"{num_nodes}",file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def save_graph_to_dataset(graph:nx.DiGraph,src_list:list):
        """
        Input:
            graph: networkx DiGraph
            src_list: source node list
        Output:
            src_list
            datastream
            trajectory_list
        """
        num_nodes=graph.number_of_nodes()
        eventstream=GraphUtils.get_eventstream(graph=graph)
        datastream=GraphUtils.compute_datastream_from_eventstream(eventstream=eventstream,num_nodes=num_nodes) # dict
        
        trajectory_list=[]
        for source_id in src_list:
            trajectory=GraphUtils.compute_TR_trajectory_from_eventstream(eventstream=eventstream,num_nodes=num_nodes,source_id=source_id) # [E,N,1]
            trajectory_list.append(trajectory)
        
        return src_list,datastream,trajectory_list

    @staticmethod
    def split_dataset(src_list:list,datastream:dict,trajectory_list:list,train_ratio:float=0.7,val_ratio:float=0.15):
        """
        Input:
        Output:
            splitted_dataset_dict:
                train: ()
                val: ()
                test: ()
                sizes: ()
        """
        E=datastream['src'].shape[0]
        train_size=int(E*train_ratio)
        val_size=int(E*val_ratio)
        test_size=E-train_size-val_size

        def _slice_ds(ds,start,end):
            return {
                'src': ds['src'][start:end],
                'tar': ds['tar'][start:end],
                'mem_t': ds['mem_t'][start:end],
                'emb_t': ds['emb_t'][start:end],
                'n_mask': ds['n_mask'][start:end]
            }

        def _slice_trajectories(traj_list,start,end):
            return [traj[start:end] for traj in traj_list]

        train_ds=_slice_ds(datastream,0,train_size)
        val_ds=_slice_ds(datastream,train_size,train_size+val_size)
        test_ds=_slice_ds(datastream,train_size+val_size,E)

        train_traj=_slice_trajectories(trajectory_list,0,train_size)
        val_traj=_slice_trajectories(trajectory_list,train_size,train_size+val_size)
        test_traj=_slice_trajectories(trajectory_list,train_size+val_size,E)

        return {
            'train': (src_list,train_ds,train_traj),
            'val': (src_list,val_ds,val_traj),
            'test': (src_list,test_ds,test_traj),
            'sizes': {'train':train_size,'val':val_size,'test':test_size}
        }
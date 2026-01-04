import os
import random
import pickle
import networkx as nx
import torch
from tqdm import tqdm
from typing_extensions import Literal
import requests
import gzip
import io

class DataUtils:
    dataset_path=os.path.join('..','data','tgnns')
    @staticmethod
    def save_to_pickle(data,file_name:str,dir_type:Literal['graph','dataset'],dataset_name:Literal['CollegeMsg','bitcoin_otc','bitcoin_alpha'],mode:Literal['train','val','test']='train'):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='dataset':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,dataset_name,mode,file_name)
        with open(file_path,'wb') as f:
            pickle.dump(data,f)
        print(f"Save {file_name}")
    
    @staticmethod
    def load_from_pickle(file_name:str,dir_type:Literal['graph','dataset'],dataset_name:Literal['CollegeMsg','bitcoin_otc','bitcoin_alpha'],mode:Literal['train','val','test']='train'):
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,dir_type,file_name)
        if dir_type=='dataset':
            file_path=os.path.join(DataUtils.dataset_path,dir_type,dataset_name,mode,file_name)
        with open(file_path,'rb') as f:
            data=pickle.load(f)
        print(f"Load {file_name}")
        return data

    @staticmethod
    def save_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.dataset_path,"inference","model",file_name)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter")

    @staticmethod
    def load_model_parameter(model,model_name:str):
        file_name=model_name+".pt"
        file_path=os.path.join(DataUtils.dataset_path,"inference","model",file_name)
        model.load_state_dict(torch.load(file_path))
        return model

    @staticmethod
    def save_memory(memory,model_name:str):
        file_name=f"{model_name}_memory.pkl"
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,"inference","memory",file_name)
        with open(file_path,'wb') as f:
            pickle.dump(memory,f)
        print(f"Save {file_name}")

    @staticmethod
    def load_memory(memory,model_name:str):
        file_name=f"{model_name}_memory.pkl"
        file_name=file_name+".pkl"
        file_path=os.path.join(DataUtils.dataset_path,"inference","memory",file_name)
        with open(file_path,'rb') as f:
            pickle.load(f)
        print(f"Load {file_name}")
        return memory

    @staticmethod
    def load_SNAP_to_graph(dataset_name:Literal['CollegeMsg','bitcoin_otc','bitcoin_alpha']='CollegeMsg',return_mapping:bool=False):
        """
        Input:
            dataset_name: SNAP dataset name
            return_mappping
        Output:
            graph: nx.DiGraph
        """
        match dataset_name:
            case 'CollegeMsg':
                snap_url=f"https://snap.stanford.edu/data/CollegeMsg.txt.gz"
            case 'bitcoin_otc':
                snap_url=f"https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
            case 'bitcoin_alpha':
                snap_url=f"https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"

        try:
            resp=requests.get(snap_url,timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Download failed: {e}")
            graph=nx.DiGraph()
            if return_mapping:
                return graph,{}
            return graph
        bio=io.BytesIO(resp.content)

        nodes=set()
        edges=[]
        with gzip.open(bio,mode='rt',encoding='utf-8',errors='ignore') as gf:
            for raw in gf:
                if raw is None:continue
                line=raw.strip()
                if not line or line.startswith('#'):continue
                # dataset-specific parsing
                match dataset_name:
                    case 'bitcoin_otc'|'bitcoin_alpha':
                        low=line.lower()
                        if low.startswith('source') or low.startswith('source,'):continue
                        parts=[p.strip().strip('"') for p in line.split(',')]
                        if len(parts)<4:continue
                        try:
                            u=int(parts[0]);v=int(parts[1]);t=float(parts[3])
                        except ValueError:continue
                    case 'CollegeMsg':
                        parts=line.split()
                        if len(parts)<3:continue
                        try:
                            u=int(parts[0]);v=int(parts[1]);t=float(parts[2])
                        except ValueError:continue
                edges.append((u,v,t))
                nodes.add(u);nodes.add(v)

        if len(edges)==0:
            print(f"zero edge events")
            graph=nx.DiGraph()
            if return_mapping:
                return graph,{}
            return graph

        mapping={orig:i for i,orig in enumerate(sorted(nodes))}
        ts=[e[2] for e in edges]
        min_t=min(ts);max_t=max(ts)
        def _scale(t):
            if max_t==min_t:return 0.6
            return 0.2+(t-min_t)/(max_t-min_t)*(1.0-0.2)

        graph=nx.DiGraph()
        graph.add_nodes_from(range(len(mapping)))
        for u,v,t in edges:
            nu=mapping[u];nv=mapping[v]
            if graph.has_edge(nu,nv):
                existing_t=graph[nu][nv].get('t',[])
                existing_t.append(_scale(t))
                graph[nu][nv]['t']=existing_t
            else:
                graph.add_edge(nu,nv,**{'t':[_scale(t)]})
        if return_mapping:
            return graph,mapping
        return graph

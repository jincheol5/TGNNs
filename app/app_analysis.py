import os
import threading
import queue
import networkx as nx
import numpy as np
import argparse
from tqdm import tqdm
from tgnns import DataUtils,GraphAnalysis,ModelTrainUtils

def app_analysis(config:dict):
    match config['app_num']:
        case 1:
            """
            App 1.
            check_elements
            """
            print(f"<<Check {config['dataset_name']} graph elements>>")
            graph=DataUtils.load_from_pickle(file_name=f"{config['dataset_name']}",dir_type="graph",dataset_name=f"{config['dataset_name']}")
            N,E_s,E=GraphAnalysis.check_elements(graph=graph)

            print(f"{config['dataset_name']} graph num_nodes: {N}")
            print(f"{config['dataset_name']} graph num_static_edgs : {E_s}")
            print(f"{config['dataset_name']} graph num_edge_events : {E}")
        
        case 2:
            """
            App 2.
            check_TR_ratio for one source
            """
            print(f"<<Check {config['dataset_name']}_{config['source_id']}_{config['mode']} TR ratio>>")
            src_list=DataUtils.load_from_pickle(file_name=f"src_list",dir_type='dataset',dataset_name=config['dataset_name'],mode=f"{config['mode']}")
            if config['source_id'] not in src_list:
                print(f"There are no source_id in src_list of {config['dataset_name']}")
                return

            traj=DataUtils.load_from_pickle(file_name=f"traj_{config['source_id']}",dir_type='dataset',dataset_name=config['dataset_name'],mode=f"{config['mode']}")
            TR_ratio=GraphAnalysis.check_tR_ratio(r=traj)
            print(f"{config['dataset_name']}_{config['source_id']}_{config['mode']} TR_ratio:{TR_ratio}")

        case 3:
            """
            App 2.
            check_TR_ratio for all source list
            """
            print(f"<<Check {config['dataset_name']}_{config['mode']} TR ratio>>")

            all_ratios=[]
            src_list=DataUtils.load_from_pickle(file_name=f"src_list",dir_type='dataset',dataset_name=config['dataset_name'],mode=f"{config['mode']}")
            for source_id in src_list:
                traj=DataUtils.load_from_pickle(file_name=f"traj_{source_id}",dir_type='dataset',dataset_name=config['dataset_name'],mode=f"{config['mode']}")
                TR_ratio=GraphAnalysis.check_tR_ratio(r=traj)
                all_ratios.append(TR_ratio)
            
            lst=np.array(all_ratios,dtype=float)
            mean_ratio=lst.mean()
            max_ratio=lst.max()
            min_ratio=lst.min()
            print(f"{config['dataset_name']}_{config['mode']} TR_ratio mean:{mean_ratio} max:{max_ratio} min:{min_ratio}")

if __name__=="__main__":
    """
    Execute app_analysis
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default="CollegeMsg")
    parser.add_argument("--source_id",type=int,default=0)
    parser.add_argument("--mode",type=str,default="train")
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'dataset_name':args.dataset_name,
        'source_id':args.source_id,
        'mode':args.mode
    }
    app_analysis(config=config)
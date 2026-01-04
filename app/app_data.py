import random
import argparse
from tqdm import tqdm
from tgnns import DataUtils,GraphUtils

def app_data(config: dict):
    match config['app_num']:
        case 1:
            """
            App 1. 
            Load SNAP dataset to graph and save using pickle.
            dataset info:
                CollegeMsg
                bitcoin_otc
                bitcoin_alpha
            """
            CollegeMsg=DataUtils.load_SNAP_to_graph(dataset_name='CollegeMsg')
            bitcoin_otc=DataUtils.load_SNAP_to_graph(dataset_name='bitcoin_otc')
            bitcoin_alpha=DataUtils.load_SNAP_to_graph(dataset_name='bitcoin_alpha')

            DataUtils.save_to_pickle(data=CollegeMsg,file_name="CollegeMsg",dir_type="graph")
            DataUtils.save_to_pickle(data=bitcoin_otc,file_name="bitcoin_otc",dir_type="graph")
            DataUtils.save_to_pickle(data=bitcoin_alpha,file_name="bitcoin_alpha",dir_type="graph")

        case 2:
            """
            App 2.
            Convert graph to dataset and save using pickle
                src_list
                datastream
                traj of each src 
            """
            graph=DataUtils.load_from_pickle(file_name=f"{config['dataset_name']}",dir_type="graph")
            node_list=list(graph.nodes())
            selected_node_list=random.sample(node_list,10) # 랜덤하게 10개의 source node 선택
            dataset=GraphUtils.convert_graph_to_dataset(graph=graph,src_list=selected_node_list)
            splitted_dataset=GraphUtils.split_dataset(dataset=dataset)

            for mode in tqdm(['train','val','test'],desc=f"convert {config['dataset_name']} to dataset and save..."):
                src_list=splitted_dataset[mode][0]
                datastream=splitted_dataset[mode][1]
                traj_list=splitted_dataset[mode][2]

                DataUtils.save_to_pickle(data=src_list,file_name=f"src_list",dir_type="dataset",dataset_name=config['dataset_name'],mode=mode)
                DataUtils.save_to_pickle(data=datastream,file_name=f"datastream",dir_type="dataset",dataset_name=config['dataset_name'],mode=mode)
                for src,traj in tqdm(zip(src_list,traj_list),desc=f"Save {config['dataset_name']} {mode} trajectory of each source..."):
                    DataUtils.save_to_pickle(data=traj,file_name=f"traj_{src}",dir_type="dataset",dataset_name=config['dataset_name'],mode=mode)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",type=str,default='CollegeMsg')
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        'dataset_name':args.dataset_name
    }
    app_data(config=config)
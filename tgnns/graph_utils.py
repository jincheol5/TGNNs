
import networkx as nx
import torch

class GraphUtils:
    @staticmethod
    def get_event_stream(graph:nx.DiGraph):
        """
        Input:
            networkx DiGraph
        Output:
            sorted tuple list: (src,tar,ts)
        """
        event_stream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                event_stream.append((int(u),int(v),float(timestamp)))
        event_stream=sorted(event_stream,key=lambda x:x[2])
        return event_stream
    
    @staticmethod
    def convert_event_stream_to_data_stream(event_stream:list,num_nodes:int):
        """
        Input:
            event_stream: sorted tuple list (src,tar,ts)
            num_nodes: number of nodes
        Output:
            data_stream: sequence of data, dict 
                data:
                    mem_t: [N,1], delta_t for memory update -> |current_time-previous_activated_time (with any nodes)| of each node
                    emb_t: [N,1], delta_t for embedding -> |current_time-previous_interaction_time (with target node)| of target node
                    src: source id
                    tar: target id
                    n_mask: [N,], neighbor mask of target node
        """

        # initial setup for delta_t in memory update, embedding
        mem_time_table=torch.zeros((num_nodes,),dtype=torch.float) # last activated time of node
        emb_time_table=torch.zeros((num_nodes,num_nodes),dtype=torch.float) # interaction time from src to tar
        
        # initial setup for n_mask
        num_edge_events=len(event_stream)
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N], 각 edge_event에 대한 tar의 neighbor mask
        neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]

        src_list=[]
        tar_list=[]
        mem_t_list=[]
        emb_t_list=[]
        n_mask_list=[]
        for idx,edge_event in enumerate(event_stream):
            src,tar,ts=edge_event
            src_list.append(src)
            tar_list.append(tar)

            activated_t=mem_time_table.unsqueeze(-1) # [N,1]
            mem_t_list.append(torch.abs(activated_t-ts))

            interacted_t=emb_time_table[tar].unsqueeze(-1) # [N,1]
            emb_t_list.append(torch.abs(interacted_t-ts))

            neighbor_history[tar][src]=True
            neighbor_mask[idx]=neighbor_history[tar] # 참조가 아닌 복사(tensor index 대입)
            n_mask_list.append(neighbor_mask[idx])

            emb_time_table[tar,src]=ts 
            mem_time_table[src]=ts
            mem_time_table[tar]=ts
        
        # convert to data_stream
        data_stream={}
        data_stream['src']=torch.tensor(src_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        data_stream['tar']=torch.tensor(tar_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        data_stream['mem_t']=torch.stack(mem_t_list,dim=0) # [E,N,1]
        data_stream['emb_t']=torch.stack(emb_t_list,dim=0) # [E,N,1]
        data_stream['n_mask']=torch.stack(n_mask_list,dim=0) # [E,N]
        return data_stream


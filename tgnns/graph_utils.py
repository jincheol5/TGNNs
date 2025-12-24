
import networkx as nx
import torch

class GraphUtils:
    @staticmethod
    def get_eventstream(graph:nx.DiGraph):
        """
        Input:
            networkx DiGraph
        Output:
            sorted tuple list: (src,tar,ts)
        """
        eventstream=[]
        for u,v,data in graph.edges(data=True):
            time_list=data['t']
            for timestamp in time_list:
                eventstream.append((int(u),int(v),float(timestamp)))
        eventstream=sorted(eventstream,key=lambda x:x[2])
        return eventstream
    
    @staticmethod
    def compute_datastream_from_eventstream(eventstream:list,num_nodes:int):
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
        num_edge_events=len(eventstream)
        neighbor_mask=torch.zeros((num_edge_events,num_nodes),dtype=torch.bool) # [E,N], 각 edge_event에 대한 tar의 neighbor mask
        neighbor_history=[torch.zeros(num_nodes,dtype=torch.bool) for _ in range(num_nodes)] # List of [N,]

        src_list=[]
        tar_list=[]
        mem_t_list=[]
        emb_t_list=[]
        n_mask_list=[]
        for idx,edge_event in enumerate(eventstream):
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
        
        # convert to datastream
        datastream={}
        datastream['src']=torch.tensor(src_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        datastream['tar']=torch.tensor(tar_list,dtype=torch.int64).unsqueeze(-1) # [E,1]
        datastream['mem_t']=torch.stack(mem_t_list,dim=0) # [E,N,1]
        datastream['emb_t']=torch.stack(emb_t_list,dim=0) # [E,N,1]
        datastream['n_mask']=torch.stack(n_mask_list,dim=0) # [E,N]
        return datastream

    @staticmethod
    def compute_TR_step(num_nodes:int,source_id:int,edge_event:tuple=None,init:bool=False,gamma:torch.Tensor=None):
        """
        Input:
            source_id
            edge_event
            init
            gamma
        Output:
            gamma
        """
        if init:
            gamma=torch.zeros((num_nodes,2),dtype=torch.float) # TR,visited_time
            gamma[:,0]=0.0 # TR
            gamma[:,1]=0.0 # visited time

            gamma[source_id,0]=1.0
            gamma[source_id,1]=0.0
        else:
            src,tar,ts=edge_event
            if gamma[src,0].item()==1.0 and gamma[tar,0].item()==0.0 and gamma[src,1].item()<ts:
                gamma[tar,0]=1.0
                gamma[tar,1]=ts
        return gamma

    @staticmethod
    def compute_TR_trajectory_from_eventstream(eventstream:list,num_nodes:int,source_id:int):
        """
        Input:
            event_stream: sorted tuple list (src,tar,ts)
            num_nodes: number of nodes
            source_id: source node id
        Output:
            TR_trajectory
        """
        TR_list=[]
        gamma=GraphUtils.compute_TR_step(num_nodes=num_nodes,source_id=source_id,init=True)
        for edge_event in eventstream:
            gamma=GraphUtils.compute_TR_step(
                num_nodes=num_nodes,
                source_id=source_id,
                edge_event=edge_event,
                gamma=gamma
            )
            TR_list.append(gamma[:,:1])
        TR_trajectory=torch.stack(TR_list,dim=0) # [E,N,1]
        return TR_trajectory
    
    
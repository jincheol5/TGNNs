import torch
import torch.nn as nn
from typing_extensions import Literal
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphSum,GraphAttention
from .model_train_utils import ModelTrainUtils

class TGAT(nn.Module):
    def __init__(self,traj_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=traj_dim+latent_dim,latent_dim=latent_dim,is_memory=False) # node_dim=traj_dim+latent_dim
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.traj_dim=traj_dim
        self.latent_dim=latent_dim

    def forward(self,data_loader,device,mode:Literal['train','test']='train'):
        """
        Input:
            data_loader: list of batch
                batch: dict
                    init_traj: [N,1]
                    traj: [B,N,1]
                    emb_t: [B,N,1]
                    mem_t: [B,N,1]
                    src: [B,1]
                    tar: [B,1]
                    n_mask: [B,N]
                    label: [B,1]
        Output:
            logit_list: List of [B,1], B는 seq 마다 크기 다를 수 있음
        """
        logit_list=[]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            batch_size,num_nodes,_=batch['trajectory'].size()
            raw=torch.zeros((batch_size,num_nodes,self.latent_dim),device=device) # [B,N,latent_dim], node raw feature
            init_traj=batch['init_traj'] # [N,1]
            init_traj=init_traj.unsqueeze(0).expand(batch_size,-1,-1) # [B,N,1]
            emb_t=batch['emb_t'] # [B,N,1]
            tar=batch['tar'] # [B,1]
            n_mask=batch['n_mask'] # [B,N]
            x=torch.cat([init_traj,raw],dim=-1) # [B,N,node_dim], node_dim=1+latent_dim 
            delta_t_vec=self.time_encoder(emb_t) # [B,N,latent_dim]
            """
            Embedding
            """
            z=self.attention(x=x,delta_t_vec=delta_t_vec,neighbor_mask=n_mask,tar_idx=tar) # [B,latent_dim]
            logit=self.linear(z) # [B,1]
            logit_list.append(logit)
        return logit_list # List of [B,1], B는 seq 마다 크기 다를 수 있음
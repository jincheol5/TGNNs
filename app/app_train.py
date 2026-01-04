import os
import random
import numpy as np
import argparse
import wandb
import torch
from tqdm import tqdm
from tgnns import DataUtils,ModelTrainer,ModelTrainUtils,TGAT,TGN

def app_train(config: dict):
    """
    seed setting
    """
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    os.environ["PYTHONHASHSEED"]=str(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic=True 
    torch.backends.cudnn.benchmark=False

    match config['app_num']:
        case 0:
            """
            App 0.
            check source list of dataset
            """
            src_list=DataUtils.load_from_pickle(file_name=f"src_list",dir_type='dataset',dataset_name=config['dataset_name'],mode='train')
            print(f"{config['dataset_name']} source list: {src_list}")

        case 1:
            """
            App 1.
            train model for one source
            """
            ### wandb
            if config['wandb']:
                if config['model']=='tgn':
                    wandb.init(project="TGNNs",name=f"{config['model']}_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}_{config['dataset_name']}_{config['source_id']}")
                else: # tgat
                    wandb.init(project="TGNNs",name=f"{config['model']}_{config['seed']}_{config['lr']}_{config['batch_size']}_{config['dataset_name']}_{config['source_id']}")
                wandb.config.update(config)

            ### data load
            src_list=DataUtils.load_from_pickle(file_name=f"src_list",dir_type='dataset',dataset_name=config['dataset_name'],mode='train')
            if config['source_id'] not in src_list:
                print(f"There are no source_id in src_list of {config['dataset_name']}")
                return
            train_datastream=DataUtils.load_from_pickle(file_name=f"datastream",dir_type='dataset',dataset_name=config['dataset_name'],mode='train')
            train_traj=DataUtils.load_from_pickle(file_name=f"traj_{config['source_id']}",dir_type='dataset',dataset_name=config['dataset_name'],mode='train')
            train_data_loader=ModelTrainUtils.get_data_loader(datastream=train_datastream,traj=train_traj,source_id=config['source_id'],batch_size=config['batch_size'])

            val_datastream=DataUtils.load_from_pickle(file_name=f"datastream",dir_type='dataset',dataset_name=config['dataset_name'],mode='val')
            val_traj=DataUtils.load_from_pickle(file_name=f"traj_{config['source_id']}",dir_type='dataset',dataset_name=config['dataset_name'],mode='val')
            val_data_loader=ModelTrainUtils.get_data_loader(datastream=val_datastream,traj=val_traj,source_id=config['source_id'],batch_size=config['batch_size'])

            ### model train
            memory=None
            match config['model']:
                case 'tgat':
                    model=TGAT(traj_dim=1,latent_dim=config['latent_dim'])
                    memory=ModelTrainer.train(model=model,is_memory=False,train_data_loader=train_data_loader,val_data_loader=val_data_loader,validate=True,config=config)
                case 'tgn':
                    model=TGN(traj_dim=1,latent_dim=config['latent_dim'],emb=config['emb'])
                    memory=ModelTrainer.train(model=model,is_memory=True,train_data_loader=train_data_loader,val_data_loader=val_data_loader,validate=True,config=config)

            ### save model
            if config['save_model']:
                match config['model']:
                    case 'tgat':
                        model_name=f"tgat_{config['seed']}_{config['lr']}_{config['batch_size']}_{config['dataset_name']}_{config['source_id']}"
                        DataUtils.save_model_parameter(model=model,model_name=model_name)
                    case 'tgn':
                        model_name=f"tgn_{config['emb']}_{config['seed']}_{config['lr']}_{config['batch_size']}_{config['dataset_name']}_{config['source_id']}"
                        DataUtils.save_model_parameter(model=model,model_name=model_name)
                        DataUtils.save_memory(memory=memory,model_name=model_name)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    
    # setting
    parser.add_argument("--model",type=str,default='tgat') # tgat,tgn
    parser.add_argument("--emb",type=str,default='attn') # time, sum, attn

    # train
    parser.add_argument("--optimizer",type=str,default='adam') # adam, sgd
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--early_stop",type=int,default=1)
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--seed",type=int,default=1) # 1, 2, 3
    parser.add_argument("--lr",type=float,default=0.0005) # 0.0005, 0,0001
    parser.add_argument("--batch_size",type=int,default=32) # 32, 64
    parser.add_argument("--latent_dim",type=int,default=32)
    
    # 학습 로그 및 저장
    parser.add_argument("--wandb",type=int,default=0)
    parser.add_argument("--save_model",type=int,default=0)

    # dataset
    parser.add_argument("--dataset_name",type=str,default='CollegeMsg')
    parser.add_argument("--source_id",type=int,default=0)
    args=parser.parse_args()

    config={
        # app 관련
        'app_num':args.app_num,
        # setting
        'model':args.model,
        'emb':args.emb,
        # train
        'optimizer':args.optimizer,
        'epochs':args.epochs,
        'early_stop':args.early_stop,
        'patience':args.patience,
        'seed':args.seed,
        'lr':args.lr,
        'batch_size':args.batch_size,
        'latent_dim':args.latent_dim,
        # 학습 로그 및 저장
        'wandb':args.wandb,
        'save_model':args.save_model,
        # dataset
        'dataset_name':args.dataset_name,
        'source_id':args.source_id
    }
    app_train(config=config)
from pytorch_lightning.loggers import WandbLogger
from data_loader import Dataloader, xlmCustomDataloader
from model import Model, xlmCustomModel

import pytorch_lightning as pl
import wandb
import argparse
import torch

sweep_config = {
    # 탐색 알고리즘 : bayesian search
    'method': 'bayes',
    # 목표달성지표 -> validation pearson maximize
    'metric': {
        'name':'val_pearson',
        'goal':'maximize'
    },
    # 찾을 hyperparameters
    'parameters': {
        'batch_size': {
            'values' : [16, 32, 64]
        },
        'loss_function': {
            'values' : ['l1_loss', 'mse_loss']
        },
        'learning_rate' :{
            'values' : [0.00001, 0.00002]
        },
        'step_size' : {
            'values' : [5, 10]
        },
        'gamma' : {
            'values' : [0.3, 0.5, 0.7]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config,
                       project='nlp_STS_sweep')

# sweep_train함수내에 모델이름은 본인에 맞춰 수정해줘야 함
def sweep_train(config=None):
    wandb.init(config=config)
    config = wandb.config

    parser = argparse.ArgumentParser()
    
    # sweep하기전 본인모델명!
    parser.add_argument('--model_name', default='xlm-roberta-large', type=str)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='../data/train_augmentation.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--custom', default=False)
    
    args = parser.parse_args(args=[])
    
    loss_fun = None
    if config.loss_function == 'l1_loss':
        loss_fun = torch.nn.L1Loss()
    elif config.loss_function == 'mse_loss':
        loss_fun = torch.nn.MSELoss()

    if args.custom & args.model_name == 'xlm-roberta-large':
        dataloader = xlmCustomDataloader(args.model_name, 
                            config.batch_size, 
                            args.shuffle, 
                            args.train_path, 
                            args.dev_path,
                            args.test_path, 
                            args.predict_path)
        model = xlmCustomModel(model_name=args.model_name,
                    lr=config.learning_rate,
                    tokenizer=dataloader.tokenizer,
                    loss_function=loss_fun,
                    step_size=config.step_size,
                    gamma=config.gamma)
    else:
        dataloader = Dataloader(args.model_name, 
                                config.batch_size, 
                                args.shuffle, 
                                args.train_path, 
                                args.dev_path,
                                args.test_path, 
                                args.predict_path)
        model = Model(model_name=args.model_name,
                    lr=config.learning_rate,
                    tokenizer=dataloader.tokenizer,
                    loss_function=loss_fun,
                    step_size=config.step_size,
                    gamma=config.gamma)
        
    wandb_logger = WandbLogger(project='nlp_STS_sweep')

    trainer = pl.Trainer(max_epochs=args.max_epoch,
                         logger=wandb_logger,
                         log_every_n_steps=1)
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

# count는 실행할 횟수설정
# 72로 설정한 이유는 현재상황에서 가능한 최대탐색횟수가 72회
    
# sweep 하다가 중단해야 하는 경우 ctrl+c 누르면 강제종료 ㄱㄴ
wandb.agent(sweep_id=sweep_id,
            function=sweep_train,
            count=3)
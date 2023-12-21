from data_loader import Dataloader, xlmCustomDataloader
from model import Model, xlmCustomModel
from text_preprocessing import TextPreprocesser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import argparse
import random

from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
import wandb

import os

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-large', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train_augmentation.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--custom', default=False)
    # loss func 후보:[L1Loss, MSELoss] -> sweep해서 나온 수치로 고쳐주기!!
    parser.add_argument('--loss_function', default=torch.nn.L1Loss())
    # lr scheduler에서 사용할 param들 -> sweep해서 나온 수치로 고쳐주기!!
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    args = parser.parse_args(args=[])
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 데이터 증강을 수행합니다.
    data_augmentation = TextPreprocesser('../data/train.csv','../data/train_augmentation.csv')
    data_augmentation.preprocessing()

    # dataloader와 model을 생성합니다.
    if args.custom and args.model_name == 'xlm-roberta-large':
        dataloader = xlmCustomDataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        model = xlmCustomModel(model_name=args.model_name, 
                               lr=args.learning_rate, 
                               tokenizer=dataloader.tokenizer,
                               loss_function=args.loss_function,
                               step_size=args.step_size,
                               gamma=args.gamma)
    else:
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        model = Model(model_name=args.model_name, 
                      lr=args.learning_rate, 
                      tokenizer=dataloader.tokenizer,
                      loss_function=args.loss_function,
                      step_size=args.step_size,
                      gamma=args.gamma)
    wandb_logger = WandbLogger(project="level1_STS",
                               name=f"batch_size:{args.batch_size}//loss_func:MSE//optim:AdamW")

    save_path = f"save_model/{args.model_name.replace('/', '_')}_Max-epoch:{args.max_epoch}_Batch-size:{args.batch_size}_custom:{args.custom}_final/"
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, 
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_pearson",
                          patience=20,
                          mode='max'),
            ModelCheckpoint(dirpath=save_path,
                            save_top_k=1,
                            monitor="val_pearson",
                            mode='max',
                            filename="best_model")
        ])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    # 학습이 완료된 모델을 저장합니다.
    trainer.save_checkpoint(save_path + "complete_model.ckpt")
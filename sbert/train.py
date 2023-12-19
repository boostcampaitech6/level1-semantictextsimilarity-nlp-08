#from kfold_data_loader import KfoldDataloader as Dataloader
from data_loader import Dataloader
from model import Model
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
    parser.add_argument('--model_name', default='jhgan/ko-sroberta-multitask')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=8e-5, type=float)
    parser.add_argument('--train_path', default='../data/converted_train.csv')
    parser.add_argument('--dev_path', default='../data/converted_dev.csv')
    parser.add_argument('--test_path', default='../data/converted_dev.csv')
    parser.add_argument('--predict_path', default='../data/converted_test.csv')
    args = parser.parse_args(args=[])
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)
    wandb_logger = WandbLogger(log_model=True)

    save_path = f"{args.model_name}_Max-epoch{args.max_epoch}_Batch-size{args.batch_size}/"
    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, 
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(dirpath=save_path,
                            save_top_k=1,
                            monitor="val_pearson",
                            mode='max',
                            filename="{epoch}-{step}-{val_pearson}")
        ])

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, 'model.pt')

import pytorch_lightning as pl
import transformers
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Subset
from sklearn.model_selection import KFold

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
    

class KfoldDataloader(pl.LightningDataModule):
    def __init__(self, 
                 model_name, 
                 batch_size, 
                 shuffle,
                 train_path, 
                 dev_path, 
                 test_path, 
                 predict_path,
                 k: int=5, # fold number(일반적으로 5 or 10)
                 split_seed: int=12345, # split needs to be always the same for correct cross validation
                 num_splits: int=10):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=128)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.setup()

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data
    
    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출하고 두 dataframe을 concat해서 total_data를 생성
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            total_data = pd.concat([train_data, val_data], axis=0)
            total_inputs, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_inputs, total_targets)

            # Dataset num_splits번 fold
            # (default) n_split: int=10, random_state: int=12345
            kf = KFold(n_splits=self.num_splits,
                       shuffle=self.shuffle,
                       random_state=self.split_seed)
            all_splits = list(kf.split(total_dataset))

            # k번째 fold된 Dataset의 index선택
            train_indices, val_indices = all_splits[ self.k ]

            # fold한 index에 따라 Dataset분할
            self.train_dataset = Subset(total_dataset, train_indices)
            self.val_dataset = Subset(total_dataset, val_indices)
            
            del total_data, total_inputs, total_targets
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
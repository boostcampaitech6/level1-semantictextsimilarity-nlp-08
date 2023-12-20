import pandas as pd
import torch
import transformers
import pytorch_lightning as pl

from tqdm.auto import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)
    

class xlmCustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs[0]
        self.source = inputs[1]
        self.targets = targets

    def __getitem__(self, idx):
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx]) , torch.tensor(self.source[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), torch.tensor(self.source[idx])

    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length = 128)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<PERSON>']})
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 모델에 맞는 sep_token으로 이어붙여서 전처리
            if self.model_name == "xlm-roberta-large":
                text = '</s>'.join([item[text_column] for text_column in self.text_columns])
            else:
                text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)
            
            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
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
    

class xlmCustomDataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length = 128)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<PERSON>']})
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.source_columns = 'source'

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '</s>'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data
    
    def one_hot_encode(self, s, source_vocab):
        one_hot_vector = [0 for _ in range(len(source_vocab))]
        one_hot_vector[source_vocab[s]] = 1
        return one_hot_vector

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except: 
            targets = []
        # 텍스트 데이터를 전처리
        inputs = self.tokenizing(data)
        
        source_s = data[self.source_columns]
        source_unique = source_s.unique()
        source_vocab = {word: idx for idx, word in enumerate(source_unique)}
        source_encoded = [self.one_hot_encode(s, source_vocab) for s in source_s]

        return inputs, targets, source_encoded

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets, train_source = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets, val_source = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용
            self.train_dataset = xlmCustomDataset([train_inputs, train_source], train_targets)
            self.val_dataset = xlmCustomDataset([val_inputs, val_source], val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets, test_source = self.preprocessing(test_data)
            self.test_dataset = xlmCustomDataset([test_inputs, test_source], test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets, predict_source = self.preprocessing(predict_data)
            self.predict_dataset = xlmCustomDataset([predict_inputs,predict_source], [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
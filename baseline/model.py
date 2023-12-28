import pytorch_lightning as pl
import transformers
import torch
import torch.nn as nn
import torchmetrics

class Model(pl.LightningModule):
    def __init__(self, model_name, lr, tokenizer, loss_function, step_size, gamma):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.loss_function = loss_function
        self.step_size = step_size
        self.gamma = gamma

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        self.plm.resize_token_embeddings(len(tokenizer))
        self.plm.config.pad_token_id

        self.loss_func = self.loss_function

    def forward(self, x):
        attention_mask = (x != 1).float()
        x = self.plm(x, attention_mask=attention_mask)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]
    

class xlmCustomModel(pl.LightningModule):
    def __init__(self, model_name, lr, tokenizer, loss_function, step_size, gamma):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.loss_function = loss_function
        self.step_size = step_size
        self.gamma = gamma
        self.hidden_size = 1024
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=self.hidden_size)
        self.plm.config.pad_token_id
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = self.loss_function
        self.plm.resize_token_embeddings(len(tokenizer))

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_size + 6, 1))

    def forward(self, x):
        attention_mask = (x != 1).float()
        x = self.plm(x, attention_mask=attention_mask)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y , s = batch
        logits = self(x)
        logits_s_concat = torch.cat((logits,s),dim=1)
        output = self.head(logits_s_concat)
        loss = self.loss_func(output, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)
        logits_s_concat = torch.cat((logits,s),dim=1)
        output = self.head(logits_s_concat)
        loss = self.loss_func(output, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(output.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        logits = self(x)
        logits_s_concat = torch.cat((logits,s),dim=1)
        output = self.head(logits_s_concat)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(output.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x, s = batch
        logits = self(x)
        logits_s_concat = torch.cat((logits,s),dim=1)
        output = self.head(logits_s_concat)
        return output.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]
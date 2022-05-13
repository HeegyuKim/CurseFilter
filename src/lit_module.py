import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.functional as tm
import pytorch_lightning as pl
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
)


def join_step_outputs(outputs):
    result = {}
    keys = outputs[0].keys()
    for k in keys:
        X = [x[k] for x in outputs]
        if X[0].dim() == 0:  # zero-dim tensor
            result[k] = torch.stack(X)
        else:
            result[k] = torch.cat(X, dim=0)
    return result


class TextClassificationModule(pl.LightningModule):
    def __init__(self, huggingface_model_name, labels, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            huggingface_model_name, num_labels=len(labels)
        )
        # config = {
        #     "max_position_embeddings": 300,
        #     "hidden_dropout_prob": 0.1,
        #     "hidden_act": "gelu",
        #     "initializer_range": 0.02, # 12 to 2
        #     "num_hidden_layers": 2,
        #     "pooler_num_attention_heads": 12,
        #     "type_vocab_size": 2,
        #     "vocab_size": 30000,
        #     "hidden_size": 128, # 768 to 128
        #     "attention_probs_dropout_prob": 0.1,
        #     "num_attention_heads": 2, # 12 to 2
        #     "intermediate_size": 512, # 3072 to 512,
        #     "num_labels": len(labels)
        # }
        # self.model = BertForSequenceClassification(
        #     BertConfig(**config)
        # )
        self.multiclass = len(labels) > 1
        self.criterion = nn.CrossEntropyLoss() if self.multiclass else nn.BCELoss()
        self.labels = labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        if self.multiclass:
            logits = logits.softmax(dim=-1)
        else:
            logits = logits.sigmoid().squeeze(1).float()
        return logits

    def training_step(self, batch, batch_idx):
        ids, masks, labels = batch

        logits = self(ids, masks)
        loss = self.criterion(logits, labels)
        output = {"loss": loss, "logits": logits, "labels": labels}
        return output

    def training_epoch_end(self, outputs):
        outputs = join_step_outputs(outputs)
        loss = outputs["loss"].mean()
        self.log("train_epoch_loss", loss)

        logits = outputs["logits"]
        labels = outputs["labels"]
        acc = tm.accuracy(logits, labels.int())
        self.log(f"train_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        ids, masks, labels = batch
        logits = self(ids, masks)
        loss = self.criterion(logits, labels)
        output = {"loss": loss, "logits": logits, "labels": labels}
        return output

    def validation_epoch_end(self, outputs):
        outputs = join_step_outputs(outputs)
        loss = outputs["loss"].mean()
        self.log("val_epoch_loss", loss, prog_bar=True)

        logits = outputs["logits"]
        labels = outputs["labels"]
        acc = tm.accuracy(logits, labels.int())
        self.log(f"val_acc", acc, prog_bar=True)


class TextClassificationStudentModule(pl.LightningModule):
    def __init__(self, config, labels, lr=5e-4, alpha=1.0):
        super().__init__()
        self.save_hyperparameters()
        if isinstance(config, str):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config, num_labels=len(labels)
            )
        else:
            self.model = BertForSequenceClassification(config)
        self.multiclass = len(labels) > 1
        self.criterion = nn.CrossEntropyLoss() if self.multiclass else nn.BCELoss()
        self.soft_label_criterion = nn.BCELoss()  # nn.KLDivLoss(reduction='batchmean')
        self.labels = labels

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt
        # sched = optim.lr_scheduler.StepLR(opt, 200, 0.5)
        # return [opt], [sched]

    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        if self.multiclass:
            logits = logits.softmax(dim=-1)
        else:
            logits = logits.sigmoid().squeeze(1).float()
        return logits

    def _shared_step(self, batch):
        ids, masks, labels, soft_labels = batch
        alpha = self.hparams.alpha

        logits = self(ids, masks)
        ce_loss = self.criterion(logits, labels)
        kd_loss = self.soft_label_criterion(logits, soft_labels)
        loss = alpha * ce_loss + (1 - alpha) * kd_loss

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "ce_loss": ce_loss,
            "kd_loss": kd_loss,
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def _shared_epoch_end(self, outputs, stage):
        outputs = join_step_outputs(outputs)
        loss_names = ["loss", "ce_loss", "kd_loss"]
        for name in loss_names:
            loss = outputs[name].mean()
            self.log(f"{stage}_epoch_{name}", loss, prog_bar=True)

        logits = outputs["logits"]
        labels = outputs["labels"]
        acc = tm.accuracy(logits, labels.int())
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

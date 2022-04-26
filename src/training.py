from typing import List, Dict, Any, Tuple
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import AutoTokenizer

from .dataset import (
    ApeachDataset,
    ApeachStudentDataset,
    DataFrameDataset,
    DataFrameStudentDataset,
)


def get_dataset(
    dataset, huggingface_model_name, max_length=64, batch_size=128, label="label"
):
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    if dataset == "apeach":
        train_ds = ApeachDataset("train", tokenizer, max_length=max_length)
        val_ds = ApeachDataset("test", tokenizer, max_length=max_length)
        label = "hate"
    elif dataset == "curse":
        train_df = pd.read_csv("data/curse_train.csv")
        val_df = pd.read_csv("data/curse_dev.csv")
        train_ds = DataFrameDataset(tokenizer, train_df, "text", "label", max_length)
        val_ds = DataFrameDataset(tokenizer, val_df, "text", "label", max_length)
        label = "curse"
    else:
        raise Exception(f"{dataset}은 지원하지 않습니다.")

    return train_ds, val_ds, [label]


def get_dataset_kd(
    dataset,
    teacher_module,
    teacher_tokenizer_name,
    student_tokenizer_name,
    max_length=64,
    label="label",
):

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_name)
    student_tokenizer = AutoTokenizer.from_pretrained(student_tokenizer_name)

    if dataset == "apeach":
        train_ds = ApeachStudentDataset(
            teacher_module, "train", teacher_tokenizer, student_tokenizer, max_length
        )
        val_ds = ApeachStudentDataset(
            teacher_module, "test", teacher_tokenizer, student_tokenizer, max_length
        )
        label = "hate"
    elif dataset == "curse":
        train_df = pd.read_csv("data/curse_train.csv")
        val_df = pd.read_csv("data/curse_dev.csv")
        train_ds = DataFrameStudentDataset(
            teacher_module,
            teacher_tokenizer,
            student_tokenizer,
            train_df,
            "text",
            "label",
            max_length,
        )
        val_ds = DataFrameStudentDataset(
            teacher_module,
            teacher_tokenizer,
            student_tokenizer,
            val_df,
            "text",
            "label",
            max_length,
        )
        label = "curse"
    else:
        raise Exception(f"{dataset}은 지원하지 않습니다.")

    return train_ds, val_ds, [label]


def train(name, module, train_dl, val_dl, conf):
    logger = pl.loggers.TensorBoardLogger(save_dir=".", name="lightning_logs")

    save_dir = "./ckpt/"
    checkpoint_callback = ModelCheckpoint(
        monitor=conf.monitor,
        dirpath=save_dir,
        filename=f"{name}_{logger.version}_" + "{val_acc:.4f}",
        mode=conf.monitor_mode,
    )

    callbacks = [
        EarlyStopping(
            conf.monitor, mode=conf.monitor_mode, patience=conf.get("patience", 3)
        ),
        checkpoint_callback,
    ]

    trainer = pl.Trainer(
        max_epochs=conf.get("max_epochs", 1000),
        logger=logger,
        gpus=1 if torch.cuda.is_available() else 0,
        val_check_interval=conf.get("val_check_interval", None),
        callbacks=callbacks,
    )
    trainer.fit(module, train_dl, val_dl)

    print(
        "Best score: ",
        checkpoint_callback.best_model_score.cpu().item(),
        "model: ",
        checkpoint_callback.best_model_path,
    )

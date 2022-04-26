import pytest
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from src.dataset import DataFrameDataset, DataFrameTSKDDataset
from src.lit_module import TextClassificationModule


max_length = 32
text_column = "text"
label_column = "label"  # label(curse_dev) bias or hate
huggingface_model_name = "beomi/kcbert-base"
csv_file = "data/curse_train.csv"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(huggingface_model_name)


@pytest.fixture
def df_dataset(tokenizer):
    dataset = DataFrameDataset(
        tokenizer,
        pd.read_csv(csv_file),
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    return dataset


def test_dataset_getitem(df_dataset: DataFrameDataset):
    ids, mask, label = df_dataset[0]
    assert ids.shape == (32,)
    assert mask.shape == (32,)
    assert label.dim() == 0


@pytest.fixture
def teacher_module():
    module = TextClassificationModule.load_from_checkpoint(
        "ckpt/curse_val_epoch_loss=0.0029.ckpt", map_location="cpu"
    )
    return module


@pytest.fixture
def tskd_dataset(teacher_module, tokenizer):
    dataset = DataFrameTSKDDataset(
        teacher_module,
        tokenizer,
        pd.read_csv(csv_file),
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    return dataset


def test_tskd_dataset_getitem(tskd_dataset: DataFrameDataset):
    ids, mask, label, soft_label = tskd_dataset[0]
    assert ids.shape == (32,)
    assert mask.shape == (32,)
    assert label.dim() == 0
    assert soft_label.dim() == 1
    print(ids, mask, label, soft_label)

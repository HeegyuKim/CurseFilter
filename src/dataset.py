from cProfile import label
from matplotlib.pyplot import text
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from datasets import load_dataset


class DataFrameDataset(Dataset):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 df: pd.DataFrame,
                 text_column: str,
                 label_column: str,
                 max_length: int = 256,
                 padding: str = "max_length") -> None:
        super().__init__()
        inputs = tokenizer(df[text_column].to_list(), padding=padding, max_length=max_length, truncation=True, return_tensors="pt")
        self.input_ids = inputs["input_ids"]
        self.attention_masks = inputs["attention_mask"]
        dtype = np.int64 if len(df[label_column].unique()) > 2 else np.float32
        self.labels = torch.from_numpy(df[label_column].values.astype(dtype))

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index: Any) -> Dict:
        return self.input_ids[index], self.attention_masks[index], self.labels[index]

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)




class DataFrameStudentDataset(DataFrameDataset):
    def __init__(self,
                 teacher_model: torch.nn.Module,
                 teacher_tokenizer: Tokenizer,
                 student_tokenizer: Tokenizer, 
                 df: pd.DataFrame,
                 text_column: str,
                 label_column: str,
                 max_length: int = 256,
                 padding: str = "max_length",
                 device: str = 'cuda') -> None:
        super().__init__(student_tokenizer, df, text_column, label_column, max_length, padding)
        
        teacher_ds = DataFrameDataset(
            teacher_tokenizer,
            df,
            text_column,
            label_column,
            max_length,
            padding
        )

        teacher_model = teacher_model.to(device)
        with torch.no_grad():
            soft_labels = [self._get_soft_label(teacher_model, teacher_ds, i, device) 
                        for i in range(len(self))]
            self.soft_labels = torch.stack(soft_labels)

    def __getitem__(self, index: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return *super().__getitem__(index), self.soft_labels[index]

    def _get_soft_label(self, model, teacher_ds, index, device):
        ids, mask, _ = teacher_ds[index]
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        return model(ids, mask).cpu().squeeze(0)


class ApeachDataset(Dataset):
    def __init__(self,
                 split: str,
                 tokenizer: Tokenizer, 
                 max_length: int = 256,
                 padding: str = "max_length") -> None:
        super().__init__()
        dataset = load_dataset("jason9693/APEACH")
        texts = dataset[split]['text']
        inputs = tokenizer(texts, padding=padding, max_length=max_length, truncation=True, return_tensors="pt")
        
        self.input_ids = inputs["input_ids"]
        self.attention_masks = inputs["attention_mask"]
        
        labels = dataset[split]['class']
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return self.input_ids.shape[0]
        
    def __getitem__(self, index: Any) -> Dict:
        return self.input_ids[index], self.attention_masks[index], self.labels[index]

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)


class ApeachStudentDataset(ApeachDataset):
    def __init__(self,
                 teacher_model: torch.nn.Module,
                 split: str,
                 teacher_tokenizer: Tokenizer, 
                 student_tokenizer: Tokenizer, 
                 max_length: int = 256,
                 padding: str = "max_length",
                 device: str="cuda") -> None:
        super().__init__(split, student_tokenizer, max_length, padding)
        
        teacher_ds = ApeachDataset(split, teacher_tokenizer, max_length, padding)

        teacher_model = teacher_model.to(device)
        with torch.no_grad():
            soft_labels = [self._get_soft_label(teacher_model, teacher_ds, i, device) 
                        for i in range(len(self))]
            self.soft_labels = torch.stack(soft_labels)

    def __getitem__(self, index: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return *super().__getitem__(index), self.soft_labels[index]

    def _get_soft_label(self, model, teacher_ds, index, device):
        ids, mask, _ = teacher_ds[index]
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        return model(ids, mask).cpu().squeeze(0)
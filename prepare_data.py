# %% [markdown]
# # Prepare Datasets
# 1. https://github.com/2runo/Curse-detection-data
# 2. https://github.com/jason9693/APEACH (Benchmark set)
# 3. https://github.com/kocohub/korean-hate-speech (BEEP!)

# %%
import os
import click

os.makedirs("raw", exist_ok=True)
os.makedirs("data/", exist_ok=True)

# %%
from urllib import request
import pandas as pd
from sklearn.model_selection import train_test_split


@click.group()
def main():
    pass


def download(url, save_path):
    request.urlretrieve(url, save_path)


def download_curse_detection():
    url = "https://raw.githubusercontent.com/2runo/Curse-detection-data/master/dataset.txt"
    download(url, "raw/curse.txt")

    return pd.read_csv("raw/curse.txt", names=["text", "label"], sep="|")


@click.command()
def prepare_curse():
    curse = download_curse_detection()
    curse = curse[curse.label.str.isdigit()].astype({"label": int})

    train, dev = train_test_split(
        curse, test_size=0.2, shuffle=True, stratify=curse.label, random_state=42
    )
    train.to_csv("data/curse_train.csv", index=False)
    dev.to_csv("data/curse_dev.csv", index=False)


main.add_command(prepare_curse, "curse")

if __name__ == "__main__":
    main()

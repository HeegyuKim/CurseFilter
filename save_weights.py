import click
import torch
import pytorch_lightning as pl
from scipy.fft import dst
from src.lit_module import TextClassificationModule, TextClassificationStudentModule


@click.command()
@click.argument("ckpt_name")
@click.argument("dst_name")
@click.argument("type")
def save_weights_model(ckpt_name: str, dst_name: str, type: str):
    print(f"Save {ckpt_name} to {dst_name} (type={type})")

    if type == "student":
        ckpt = TextClassificationStudentModule.load_from_checkpoint(ckpt_name)
    else:
        ckpt = TextClassificationModule.load_from_checkpoint(ckpt_name)

    torch.save(
        {
            "state_dict": ckpt.state_dict(),
            pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY: dict(ckpt.hparams),
        },
        dst_name,
    )

    # test
    if type == "student":
        ckpt = TextClassificationStudentModule.load_from_checkpoint(dst_name)
    else:
        ckpt = TextClassificationModule.load_from_checkpoint(dst_name)


if __name__ == "__main__":
    save_weights_model()

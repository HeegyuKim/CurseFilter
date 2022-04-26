import hydra
import click
from omegaconf import DictConfig, OmegaConf

from src.lit_module import TextClassificationModule
from src.training import get_dataset, train


@click.command()
@click.argument("yaml")
def main(yaml):
    hydra.initialize("conf", job_name="curse_hate_filter")
    conf = hydra.compose(yaml)
    print(OmegaConf.to_yaml(conf))

    train_ds, val_ds, labels = get_dataset(
        dataset=conf.dataset,
        huggingface_model_name=conf.model.huggingface_model_name,
        max_length=conf.model.max_length,
        label=conf.model.label,
    )

    module = TextClassificationModule(
        conf.model.huggingface_model_name, labels=labels, lr=conf.training.lr
    )

    train(
        conf.dataset,
        module,
        train_ds.dataloader(batch_size=conf.training.batch_size),
        val_ds.dataloader(batch_size=conf.training.batch_size),
        conf.training,
    )


if __name__ == "__main__":
    main()

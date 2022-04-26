import hydra
import click
from omegaconf import DictConfig, OmegaConf

from src.lit_module import TextClassificationModule, TextClassificationStudentModule
from src.training import get_dataset_kd, train


@click.command()
@click.argument("yaml")
def main(yaml):
    hydra.initialize("conf", job_name="curse_hate_filter")
    conf = hydra.compose(yaml)
    print(OmegaConf.to_yaml(conf))

    teacher = TextClassificationModule.load_from_checkpoint(
        conf.model.teacher.checkpoint
    )

    train_ds, val_ds, labels = get_dataset_kd(
        conf.dataset,
        teacher,
        conf.model.teacher.tokenizer,
        conf.model.student.huggingface_model_name,
        max_length=conf.model.student.max_length,
        label=conf.model.student.label,
    )

    module = TextClassificationStudentModule(
        conf.model.student.huggingface_model_name,
        labels=labels,
        lr=conf.training.lr,
        alpha=conf.model.kd.alpha,
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

from datasets import load_dataset
from targe.conversation import format_data


def load_chartqa(cfg):
    dataset_id = cfg.get("dataset_id", "HuggingFaceM4/ChartQA")
    splits = list(cfg.get("splits", ["train", "val", "test"]))
    train_ds, eval_ds, test_ds = load_dataset(dataset_id, split=splits)
    return (
        [format_data(s) for s in train_ds],
        [format_data(s) for s in eval_ds],
        [format_data(s) for s in test_ds],
    )

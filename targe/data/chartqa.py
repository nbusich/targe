from datasets import load_dataset
from targe.conversation import format_data


def load_chartqa(cfg):
    num_proc = cfg.get('dataloader_num_workers', 0)
    dataset_id = cfg.get("dataset_id", "HuggingFaceM4/ChartQA")
    splits = list(cfg.get("splits", ["train", "val", "test"]))
    train_ds, val_ds, test_ds = load_dataset(dataset_id, split=splits)
    train_ds = train_ds.map(format_data, batched=False, num_proc=num_proc,desc="Formatting train split")
    val_ds = val_ds.map(format_data, num_proc=num_proc, desc="Formatting val split")
    test_ds = test_ds.map(format_data, num_proc=num_proc, desc="Formatting test split")
    return (train_ds, val_ds, test_ds)

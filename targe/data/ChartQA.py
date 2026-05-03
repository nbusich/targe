from datasets import load_dataset

from targe.conversation import format_data

DATASET_ID = "HuggingFaceM4/ChartQA"
DEFAULT_SPLITS = ("train[:10%]", "val[:10%]", "test[:10%]")


def load_chartqa(splits=DEFAULT_SPLITS, dataset_id: str = DATASET_ID):
    train_ds, eval_ds, test_ds = load_dataset(dataset_id, split=list(splits))
    return (
        [format_data(s) for s in train_ds],
        [format_data(s) for s in eval_ds],
        [format_data(s) for s in test_ds],
    )

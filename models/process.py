import json
from models.config import Config, TrainInput
from models.utils.uie_tokenizer import T5BertTokenizer
from models.utils.meta_data_collator import DataCollatorForMetaSeq2Seq
import datasets
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, config: Config, train_input: TrainInput, name: str):
        super(MyDataset).__init__()
        paths = {'train': config.train_file, 'val': config.validation_file, 'test': config.test_file}
        dataset = self.read_raw_data(paths[name])
        dataset = self.preprocess_function(dataset, train_input.tokenizer, config)
        self.dataset = datasets.Dataset.from_dict(dataset)

    @staticmethod
    def preprocess_function(examples, tokenizer: T5BertTokenizer, config: Config):
        inputs = examples[config.text_column]
        targets = examples[config.record_column]
        inputs = [config.prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=config.max_source_length, padding="max_length", truncation=True).__dict__["data"]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=config.max_target_length, padding="max_length", truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if config.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in
                labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        if config.source_prefix is not None and config.source_prefix.startswith('meta'):
            model_inputs['spots'] = examples['spot']
            model_inputs['asocs'] = examples['asoc']
            model_inputs['spot_asoc'] = examples['spot_asoc']
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        return model_inputs

    def __getitem__(self, item: int):
        json_data = self.dataset[item]
        return json_data

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def read_raw_data(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
        raw_data = {}
        for i in data:
            data = json.loads(i)
            for key, val in data.items():
                if key not in raw_data:
                    raw_data[key] = []
                raw_data[key].append(val)
        return raw_data


class CollateFn:
    def __init__(self, config: Config, train_input: TrainInput):
        self.collate_fn = DataCollatorForMetaSeq2Seq(
            train_input.tokenizer,
            label_pad_token_id=-100 if config.ignore_pad_token_for_loss else train_input.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if config.fp16 else None,
            max_length=config.max_source_length,
            max_prefix_length=config.max_prefix_length,
            max_target_length=config.max_target_length,
            negative_sampler=train_input.negative_sampler,
            spot_asoc_nosier=train_input.spot_asoc_nosier,
            decoding_format=config.decoding_format,
        )


def get_data_iterator(config: Config, train_input: TrainInput, name: str):
    dataset = MyDataset(config, train_input, name)
    return DataLoader(
        dataset=dataset,
        collate_fn=CollateFn(config, train_input).collate_fn,
        batch_size=config.train_batch_size)

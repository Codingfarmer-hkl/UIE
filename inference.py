#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re
from tqdm import tqdm
from models.utils.scorer import EntityScorer, RelationScorer, EventScorer
from models.utils.record_schema import RecordSchema
from models.utils.record import MapConfig
from models.utils.sel2record import SEL2Record
from models.utils.uie_tokenizer import T5BertTokenizer
from models.uie import UieModelInference
import math
import os
import torch
import argparse

split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}


def read_json_file(file_name):
    return [json.loads(line) for line in open(file_name, "r", encoding="utf-8")]


def schema_to_ssi(schema: RecordSchema):
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi


def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()


class Predictor:
    def __init__(self, model: UieModelInference, tokenizer: T5BertTokenizer, record_schema: RecordSchema,
                 args: argparse) -> None:
        self._tokenizer = tokenizer
        self._model = model
        # self._model.cuda()
        self._schema = record_schema
        self._ssi = schema_to_ssi(self._schema)
        self.device = args.device
        self._max_source_length = args.max_source_length
        self._max_target_length = args.max_target_length

    def predict(self, text):
        text = [self._ssi + x for x in text]
        inputs = self._tokenizer(
            text, padding=True, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length]
        result = self._model(input_ids=inputs['input_ids'].to(self.device), attention_mask=inputs['attention_mask'].to(self.device), max_target_length=self._max_target_length)
        return self._tokenizer.batch_decode(result, skip_special_tokens=False, clean_up_tokenization_spaces=False)


task_dict = {
    'entity': EntityScorer,
    'relation': RelationScorer,
    'event': EventScorer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        '--data_folder', default='data/train_data')
    parser.add_argument(
        '--test_data', default='data/temp_data/train.json')
    parser.add_argument(
        '--model_path', default='model_weights/pre-training-uie-char-small')
    parser.add_argument(
        '--test_result', default='data/test_result')
    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=256, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--config', dest='map_config',
                        help='Offset Re-mapping Config',
                        default='models/closest_offset_zh.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--record_schema_path', type=str, default='data/train_data/record.schema')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal',
                        choices=['set', 'normal', 'multimatch'])
    args = parser.parse_args()

    tokenizer = T5BertTokenizer.from_pretrained(args.model_path)
    model = UieModelInference(args.model_path, args.device)
    record_schema = RecordSchema.read_from_file(args.record_schema_path)
    predictor = Predictor(model, tokenizer, record_schema, args)

    map_config = MapConfig.load_from_yaml(args.map_config)
    schema_dict = SEL2Record.load_schema_dict(args.data_folder)
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=args.decoding,
        map_config=map_config,
    )

    # for split, split_name in [('val', 'eval'), ('test', 'test')]:
    #     gold_filename = f"{data_folder}/{split}.json"

    text_list = [x['text'] for x in read_json_file(args.test_data)]
    token_list = [x['tokens'] for x in read_json_file(args.test_data)]
    batch_num = math.ceil(len(text_list) / args.batch_size)

    predict = list()
    for index in tqdm(range(batch_num)):
        start = index * args.batch_size
        end = index * args.batch_size + args.batch_size
        pred_seq2seq = predictor.predict(text_list[start: end])
        pred_seq2seq = [post_processing(x) for x in pred_seq2seq]
        predict += pred_seq2seq

    records = list()
    for p, text, tokens in zip(predict, text_list, token_list):
        r = sel2record.sel2record(pred=p, text=text, tokens=tokens)
        records += [r]

    results = dict()
    for task, scorer in task_dict.items():
        gold_list = [x[task] for x in read_json_file(args.test_data)]
        pred_list = [x[task] for x in records]

        gold_instance_list = scorer.load_gold_list(gold_list)
        pred_instance_list = scorer.load_pred_list(pred_list)

        sub_results = scorer.eval_instance_list(
            gold_instance_list=gold_instance_list,
            pred_instance_list=pred_instance_list,
            verbose=args.verbose,
            match_mode=args.match_mode,
        )
        results.update(sub_results)

        with open(os.path.join(args.test_result, 'test_preds_record.txt'), 'w', encoding="utf-8") as output:
            for record in records:
                output.write(f'{json.dumps(record)}\n')

        with open(os.path.join(args.test_result, 'test_preds_seq2seq.txt'), 'w', encoding="utf-8") as output:
            for pred in predict:
                output.write(f'{pred}\n')

        with open(os.path.join(args.test_result, 'test_results.txt'), 'w', encoding="utf-8") as output:
            for key, value in results.items():
                output.write(f"{'test'}_{key}={value}\n")


if __name__ == "__main__":
    main()

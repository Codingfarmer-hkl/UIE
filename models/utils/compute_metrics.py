from typing import List
from models.utils.record_schema import RecordSchema
from models.utils.scorer import Metric, RecordMetric, OrderedRecordMetric
import numpy as np
from models.utils.spotasoc_predict_parser import SpotAsocPredictParser
from models.config import Config, TrainInput


def eval_pred(predict_parser: SpotAsocPredictParser, gold_list, pred_list, text_list=None, raw_list=None):
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list
    )

    spot_metric = Metric()
    asoc_metric = Metric()
    record_metric = RecordMetric()
    ordered_record_metric = OrderedRecordMetric()

    for instance in well_formed_list:
        spot_metric.count_instance(instance['gold_spot'], instance['pred_spot'])
        asoc_metric.count_instance(instance['gold_asoc'], instance['pred_asoc'])
        record_metric.count_instance(instance['gold_record'], instance['pred_record'])
        ordered_record_metric.count_instance(instance['gold_record'], instance['pred_record'])

    spot_result = spot_metric.compute_f1(prefix='spot-')
    asoc_result = asoc_metric.compute_f1(prefix='asoc-')
    record_result = record_metric.compute_f1(prefix='record-')
    ordered_record_result = ordered_record_metric.compute_f1(prefix='ordered-record-')

    overall_f1 = spot_result.get('spot-F1', 0.) + asoc_result.get('asoc-F1', 0.)
    # print(counter)
    result = {'overall-F1': overall_f1}
    result.update(spot_result)
    result.update(asoc_result)
    result.update(record_result)
    result.update(ordered_record_result)
    result.update(counter)
    return result


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: RecordSchema):
    predict_parser = SpotAsocPredictParser(label_constraint=label_constraint)  # 实体解析，从预测结果中解析实体
    return eval_pred(
        predict_parser=predict_parser,
        gold_list=tgt_lns,
        pred_list=pred_lns
    )


def postprocess_text(x_str: str, to_remove_token_list: list):
    # Clean `bos` `eos` `pad` for cleaned text
    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, '')

    return x_str.strip()


def compute_metrics(decoded_preds, decoded_labels, config: Config, train_input: TrainInput):
    # if isinstance(preds, tuple):
    #     preds = preds[0]
    # decoded_preds = train_input.tokenizer.batch_decode(preds, skip_special_tokens=False,
    #                                                    clean_up_tokenization_spaces=False)  # word id 转word
    # if config.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, train_input.tokenizer.pad_token_id)
    # decoded_labels = train_input.tokenizer.batch_decode(labels, skip_special_tokens=False,
    #                                                     clean_up_tokenization_spaces=False)  # word id 转word

    decoded_preds = [postprocess_text(x, config.to_remove_token_list) for x in decoded_preds]  # 去除一些不正确的标识符
    decoded_labels = [postprocess_text(x, config.to_remove_token_list) for x in decoded_labels]  # 去除一些不正确的标识符

    result = get_extract_metrics(
        pred_lns=decoded_preds,
        tgt_lns=decoded_labels,
        label_constraint=train_input.record_schema  # 训练的实体类型和关系类型
    )

    # prediction_lens = [np.count_nonzero(pred != train_input.tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

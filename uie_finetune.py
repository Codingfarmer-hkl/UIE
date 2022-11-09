from models.uie import UieModelTrain
from models.process import get_data_iterator
from models.utils.uie_tokenizer import T5BertTokenizer
import torch.optim as optim
from models.utils.compute_metrics import compute_metrics
from models.utils.earlystopping import EarlyStopping
from models.utils.meta_data_collator import DynamicSSIGenerator
from models.utils.spot_asoc_noiser import SpotAsocNoiser
from models.utils.record_schema import RecordSchema
from models.config import Config, TrainInput
from tqdm import tqdm

def train(model, data_loader, optimizer, device):
    model.train()
    for inputs in tqdm(data_loader):
        # prepare decoder_input_ids
        if model is not None and hasattr(model.uie_model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.uie_model.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
            inputs["decoder_input_ids"] = decoder_input_ids.to(device)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        inputs["labels"] = inputs["labels"].to(device)
        outputs = model(inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validation(model, data_loader, device):
    model.eval()
    loss = 0
    for i, inputs in enumerate(data_loader):
        if model is not None and hasattr(model.uie_model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.uie_model.prepare_decoder_input_ids_from_labels(labels=inputs["labels"])
            inputs["decoder_input_ids"] = decoder_input_ids.to(device)
        outputs = model(inputs)
        loss += outputs[0]
    return loss


def predict(model, data_loader, config: Config, train_input: TrainInput):
    model.eval()
    pred_labels = []
    real_labels = []
    import numpy as np
    for i, inputs in enumerate(data_loader):
        labels = inputs['labels']
        outputs = model.uie_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                           max_length=config.max_target_length)
        decoded_preds = train_input.tokenizer.batch_decode(outputs, skip_special_tokens=False,
                                                           clean_up_tokenization_spaces=False)  # word id 转word
        if config.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, train_input.tokenizer.pad_token_id)
        decoded_labels = train_input.tokenizer.batch_decode(labels, skip_special_tokens=False,
                                                            clean_up_tokenization_spaces=False)  # word id 转word
        pred_labels += decoded_preds
        real_labels += decoded_labels
    result = compute_metrics(pred_labels, real_labels, config, train_input)
    print(result)

    return result


def main():
    config = Config()  # 算法参数
    tokenizer = T5BertTokenizer.from_pretrained(config.model_path)
    to_remove_token_list = list()  # 获取特殊字符，用于去除预测生成的数据中无用字符
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:  # eos_token <\s>
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:  # pad_token <pad>
        to_remove_token_list += [tokenizer.pad_token]

    config.to_remove_token_list = to_remove_token_list
    record_schema = RecordSchema.read_from_file(config.record_schema_path)  # 读取record_schema及训练的所有的实体和关系
    negative_sampler = DynamicSSIGenerator(tokenizer, record_schema, config.meta_positive_rate, config.meta_negative,
                                           config.ordered_prompt)  # 动态生成SSI负样本
    spot_asoc_nosier = SpotAsocNoiser(spot_noise_ratio=config.spot_noise,
                                      asoc_noise_ratio=config.asoc_noise)  # 控制加入噪声数据的比例
    early_stopping = EarlyStopping(patience=config.patience, delta=config.delta,
                                   save_path=config.output_dir)  # 控制训练提前停止
    train_input = TrainInput(record_schema=record_schema, negative_sampler=negative_sampler,
                             spot_asoc_nosier=spot_asoc_nosier, tokenizer=tokenizer, early_stopping=early_stopping)

    train_data_iterator = get_data_iterator(config, train_input, "train")  # 获取训练迭代器
    test_data_iterator = get_data_iterator(config, train_input, "test")
    val_data_iterator = get_data_iterator(config, train_input, "val")

    model = UieModelTrain(config.model_path).to(config.device)  # 模型初始化

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # 训练
    for epoch in range(config.epochs):
        train(model, train_data_iterator, optimizer, config.device)
        loss = validation(model, val_data_iterator, config.device)
        early_stopping(loss, model, tokenizer)
        if early_stopping.early_stop:
            break
    predict(model, test_data_iterator, config, train_input)




if __name__ == "__main__":
    main()

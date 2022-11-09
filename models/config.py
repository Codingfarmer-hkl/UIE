from models.utils.record_schema import RecordSchema
from dataclasses import dataclass
from models.arg_utils import fetch_args
from models.utils.uie_tokenizer import T5BertTokenizer
from models.utils.earlystopping import EarlyStopping
from models.utils.meta_data_collator import DynamicSSIGenerator
from models.utils.spot_asoc_noiser import SpotAsocNoiser
import torch
args = fetch_args()

@dataclass
class Config:
    """
    model_path:Path to pretrained model or model identifier from huggingface.co/models
    text_column:The name of the column in the datasets containing the full texts
    """
    train_path:str = "temp_data"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    to_remove_token_list: list = None
    model_path: str = 'model_weights/pre-training-uie-char-small'
    task: str = 'meta'
    text_column: str = 'text'
    record_column: str = 'record'
    train_file: str = f'data/{train_path}/train.json'
    validation_file: str = f'data/{train_path}/val.json'
    test_file: str = f'data/{train_path}/test.json'
    max_source_length: int = 1024
    max_prefix_length: int = None
    max_target_length: int = 128
    pad_to_max_length: bool = False
    ignore_pad_token_for_loss: str = True
    source_prefix: str = 'meta'
    meta_negative: int = -1
    ordered_prompt: bool = True
    decoding_format: str = 'spotasoc'
    record_schema_path: str = f'data/{train_path}/record.schema'
    spot_noise: float = 0.1
    asoc_noise: float = 0.1
    meta_positive_rate: float = 0.991
    output_dir: str = 'model_weights/fine-tuned-weights'
    train_batch_size: int = 3
    per_batch_size: int = 3
    fp16: bool = False
    lr: float = 0.0001
    epochs: int = 3
    patience: int = 5
    delta: float = 0.001
    prefix: str = str()



@dataclass
class TrainInput:
    record_schema: RecordSchema = None
    negative_sampler: DynamicSSIGenerator = None
    spot_asoc_nosier: SpotAsocNoiser = None
    tokenizer: T5BertTokenizer = None
    early_stopping: EarlyStopping = None



import argparse


def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model/pre-training-uie-char-small",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument('--task', type=str, default="meta",
                        help="The name of the task, should be summarization (or summarization_{dataset} for evaluating pegasus) or translation (or translation_{xx}_to_{yy}).")
    parser.add_argument('--text_column', type=str, default='text',
                        help="The name of the column in the datasets containing the full texts (for summarization).", )
    parser.add_argument('--record_column', type=str, default='record',
                        help="The name of the column in the datasets containing the summaries (for summarization).", )
    parser.add_argument('--train_file', type=str, default="train_data/train.json",
                        help="The input training train_data file (a jsonlines or csv file).")
    parser.add_argument('--validation_file', type=str, default="train_data/val.json",
                        help="An optional input evaluation train_data file to evaluate the metrics (rouge/sacreblue) on (a jsonlines or csv file).")
    parser.add_argument('--test_file', type=str, default="train_data/test.json",
                        help="An optional input test train_data file to evaluate the metrics (rouge/sacreblue) on (a jsonlines or csv file).")
    parser.add_argument('--max_source_length', type=int, default=1024,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_prefix_length', type=int, default=None, help="The maximum prefix length.")
    parser.add_argument('--max_target_length', type=int, default=128,
                        help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--pad_to_max_length', type=bool, default=False,
                        help="Whether to pad all samples to model maximum sentence length.If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU.")
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True,
                        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.")
    parser.add_argument('--source_prefix', type=str, default="meta",
                        help="A prefix to add before every source text (useful for T5 models).")
    parser.add_argument('--meta_negative', type=int, default=-1, help="Negative Schema Number in Training.")
    parser.add_argument('--ordered_prompt', type=bool, default=True,
                        help="Whether to sort the spot prompt and asoc prompt or not.")
    parser.add_argument('--decoding_format', type=str, default='spotasoc', help="Decoding Format")
    parser.add_argument('--record_schema', type=str, default="train_data/record.schema", help="The input event schema file.")
    parser.add_argument('--spot_noise', type=float, default=0.1, help="The noise rate of null spot.")
    parser.add_argument('--asoc_noise', type=float, default=0.1, help="The noise rate of null asoc.")
    parser.add_argument('--meta_positive_rate', type=float, default=0.991, help="The keep rate of positive spot.")
    parser.add_argument('--fine-tuned-weights', type=str, default="fine-tuned-weights",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_batch_size', type=int, default=3, help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument('--per_batch_size', type=int, default=1, help="Batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument('--fp16', type=bool, default=False,
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument('--lr', type=float, default=10e-5, help="learning_rate")
    parser.add_argument('--epochs', type=int, default=1, help="")
    parser.add_argument('--patience', type=int, default=5, help="How long to wait after last time validation loss improved")
    parser.add_argument('--delta', type=int, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement")

    args = parser.parse_args()
    return args

from torch import nn
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


class UieModelTrain(nn.Module):
    def __init__(self, model_path):
        super(UieModelTrain, self).__init__()
        self.uie_model = T5ForConditionalGeneration.from_pretrained(model_path)
        for params in self.uie_model.parameters():
            params.requires_grad = True

    def forward(self, inputs):
        output = self.uie_model(**inputs)
        return output


class UieModelInference(nn.Module):
    def __init__(self, inference_model_path, device):
        super(UieModelInference, self).__init__()
        self.uie_model = T5ForConditionalGeneration.from_pretrained(inference_model_path).to(device)
        for params in self.uie_model.parameters():
            params.requires_grad = True

    def forward(self, input_ids, attention_mask, max_target_length):
        output = self.uie_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                         max_length=max_target_length)
        return output

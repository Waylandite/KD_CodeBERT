import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = x.reshape(-1, x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder,fit_size=768,block_size=400):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = RobertaClassificationHead(encoder.config)
        self.fit_dense = nn.Linear(encoder.config.hidden_size, fit_size)
        self.block_size=block_size

    def forward(self, input_ids=None,mod="train", is_student=False):
        input_ids = input_ids.view(-1, self.block_size)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1),output_attentions=True,
                               output_hidden_states=True,
                               return_dict=True)
        last_hidden_state = outputs.get('last_hidden_state')
        logits = self.classifier(last_hidden_state)
        outputs['logits'] = logits
        if is_student:
            sequence_output = outputs.get('hidden_states')
            tmp = []
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
            outputs['hidden_states'] = sequence_output
        if mod == "train":
            return outputs
        else:
            return logits

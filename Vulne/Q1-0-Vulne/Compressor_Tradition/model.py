import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder,fit_size=768):
        super(Model, self).__init__()
        self.encoder = encoder
        self.fit_dense = nn.Linear(encoder.config.hidden_size, fit_size)
        
    def forward(self, input_ids=None,mod="train",is_student=False):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1), output_attentions=True,
                               output_hidden_states=True,
                               return_dict=True)
        logits = outputs.get('logits')
        # outputs在train时使用，logits在eval时使用
        if is_student:
            sequence_output=outputs.get('hidden_states')
            tmp = []
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
            outputs['hidden_states']=sequence_output
        if mod == "train":
            return outputs
        else:
            return logits

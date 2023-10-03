import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07)), model_id=None):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        self.model_id = model_id
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def __repr__(self):
        default_text = super().__repr__()
        if self.model_id is not None:
            custom_text = f'model_id: {self.model_id} '
        else:
            custom_text = ''
        return custom_text + default_text
    
    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x
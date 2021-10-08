import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ConfidenceModule(nn.Module):
    def __init__(self, conf_model, conf_type='softmax'):
        super().__init__()
        assert conf_type in ['linear', 'softmax'] 
        self.conf_model = conf_model
        self.conf_type = conf_type
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x, extra_mask=None, temperature=1, return_logit=False):

        if extra_mask is None:
            extra_mask = torch.ones_like(x)

        if self.conf_type == 'linear':
            conf = (F.elu(self.conf_model(x))+1+1e-12) * (extra_mask+1e-12)
        elif self.conf_type == 'softmax':
            # conf = self.conf_model(x)
            logit = self.conf_model(x)
            # conf = torch.where(extra_mask > 0,
            #                         conf, torch.full_like(conf, -1e20))
            # min_conf = torch.min(conf)
            conf = torch.where(extra_mask > 0,
                                    logit, torch.full_like(logit, -1000))

            conf_shape = conf.shape
            conf = conf.reshape(*conf.shape[0:2], -1)

            conf = self.softmax(conf/temperature)
            conf = conf.reshape(*conf_shape)#.clone()
        if return_logit:
            return conf, logit
        else:
            return conf
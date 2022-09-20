import torch.nn as nn
import torch.nn.functional as F
import torch

def stable_sigmoid(logits):
    """
    logits : B * C * H * W
    """
    device = logits.device
    one = torch.tensor([1.0],dtype = torch.float32,device = device)
    preds = torch.where(logits > 0.0,1.0 / (one + (-logits).exp()),
            logits.exp() / (one + logits.exp()))
    
    return preds

class reweighted_Cross_Entroy_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,preds,targets):
        """
        preds : [B,K,H,W]  predicted binary edge map for K classes,belong to [0,1]
        targets : [B,K,H,W] standard binary edge map for K classes,belong to {0,1}
        """
        device = preds.device
        B,K,H,W = preds.shape
        # preds = stable_sigmoid(logits)
        

        loss_total = torch.tensor([0.0],dtype = torch.float32,device = device)
        # B,K,H,W = preds.shape
        num_total = B * H * W
        for i in range(K): # iterate for batch size
            pred = preds[:,i,...]
            target = targets[:,i,...]

            num_pos = torch.sum(target) # true positive number
            # if num_pos == 0:
            #     continue
            num_neg = num_total - num_pos
            pos_weight = (num_neg / num_pos).clamp(min = 1.0,max = num_total) # compute a pos_weight for each image
            # neg_weight = num_pos / num_total

            # loss = (pos_weight * target * pred + \
            #         neg_weight * (1 - target) * (1 - pred))

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = pred - pred * target + log_weight * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())

            loss = loss.mean()
            loss_total = loss_total + loss

        loss_total = loss_total /  targets.size(1)
        return loss_total
        



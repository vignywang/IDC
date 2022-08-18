import torch
import torch.nn as nn
import torch.nn.functional as F
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()
    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim

def kurtosis_torch(x, dim=(2, 3), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis.squeeze(-1).squeeze(-1).squeeze(-1)
    return kurtosis

class Cls_loss(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

class Con_loss(nn.Module):
    def __init__(self,loc_weight=2,global_weight=1):
        super().__init__()
        self.lw = loc_weight
        self.gw = global_weight
    def forward(self, fmap1, fmap2):
        ns, cs, hs, ws = fmap1.size()
        fmap1 = F.sigmoid(fmap1)
        fmap2 = F.sigmoid(fmap2)
        l_loss = torch.abs(fmap1 - fmap2).cuda()
        l_loss = torch.mean(torch.topk(l_loss.view(ns, -1), k=(int)(hs * ws * 0.2), dim=-1)[0]) # top 20% hard pixels
        k1 = kurtosis_torch(fmap1)
        k2 = kurtosis_torch(fmap2)
        g_loss = torch.mean(torch.abs(k1 - k2).cuda()) # kurtosis Consistency
        return self.lw*l_loss+self.gw*g_loss



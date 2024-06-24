import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SoftTargetInfoNCE(nn.Module):
    def __init__(self, noise_probs, t=1.0):
        super(SoftTargetInfoNCE, self).__init__()
        self.t = t 
        self.log_noise_probs = torch.log(noise_probs)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (N,) embeddings from query encoder
            targets: (N, num_classes) (soft) targets - weights for each class
        """
        # check if process group is initialized
        if torch.distributed.is_initialized():
            targets = concat_all_gather(targets)
        logits = (logits / self.t) - self.log_noise_probs.unsqueeze(0)
        # logits = (logits / self.t) - torch.log(self.eta).unsqueeze(0)

        N = logits.shape[0]  # batch size
        M = targets.shape[0]  # usually M > N
        d = logits.shape[1]  # num_classes

        w = targets.repeat(N,1).unsqueeze(1).reshape(N, M, d)
        logits = (logits.unsqueeze(1) * w).sum(-1)
 
        # for numerical stability
        logits = logits - (torch.max(logits, dim=1, keepdim=True)[0]).detach()
        
        if torch.distributed.is_initialized():
            labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        else:
            labels = torch.arange(N, dtype=torch.long).cuda()

        return nn.CrossEntropyLoss()(logits, labels)


class InfoNCE(SoftTargetInfoNCE):
    def __init__(self, eta, t=1.0):
        super().__init__(eta, t)
    
    def forward(self, logits, labels):
        targets = torch.nn.functional.one_hot(labels, num_classes=logits.size(1))
        return super().forward(logits, targets)


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, t):
        super(SoftTargetCrossEntropy, self).__init__()
        self.t = t 

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = x / self.t
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
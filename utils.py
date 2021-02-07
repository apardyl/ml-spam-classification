import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def load_train_state(file_path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    if 'scheduler' in data:
        scheduler.load_state_dict(data['scheduler'])
    return data['epoch'], data.get('best_score'), data.get('vocabulary')


def load_classifier(file_path, model: nn.Module):
    data = torch.load(file_path)
    model.load_state_dict(data['model_state'])
    return data.get('vocabulary')


def save_train_state(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, best_score: float,
                     vocabulary, file_path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_score': best_score,
        'vocabulary': vocabulary,
    }, file_path)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, X, target):
        if X.dim() > 2:
            X = X.view(X.size(0), X.size(1), -1)
            X = X.transpose(1, 2)
            X = X.contiguous().view(-1, X.size(2))
        target = target.view(-1, 1)

        log_pt = F.log_softmax(X, dim=-1)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != X.data.type():
                self.alpha = self.alpha.type_as(X.data)
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

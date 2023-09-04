import torch
import numpy as np


@torch.no_grad()
def knn_evaluation(val_loader, model, memory_bank, device):
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    model.eval()

    for Xs, target in val_loader:
        Xs, target = [x.to(device) for x in Xs], target.to(device)

        output = model.consistency_repr(Xs, normlization=True)
        total_num += output.size(0)
        output = memory_bank.weighted_knn(output) 
        
        total_top1 += torch.sum((output[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_top5 += torch.sum((output[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # print(output.unique())
        # total_top1 = 100*torch.mean(torch.eq(output, target).float()).item()

    total_top1 = (total_top1 / total_num) * 100
    total_top5 = (total_top5 / total_num) * 100
    return {'knn_acc@1': total_top1, 'knn_acc@5': total_top5}


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, device):
    model.eval()
    memory_bank.reset()

    for i, (Xs, targets) in enumerate(loader):
        Xs, targets = [x.to(device) for x in Xs], targets.to(device)
        output = model.consistency_repr(Xs, normlization=True)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
            
            
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def label_to_one_hot(label_idx, num_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(label_idx, 
                                       num_classes=num_classes)
    
def one_hot_to_label(one_hot_arr: torch.Tensor) -> torch.Tensor:
    return one_hot_arr.argmax(dim=1)
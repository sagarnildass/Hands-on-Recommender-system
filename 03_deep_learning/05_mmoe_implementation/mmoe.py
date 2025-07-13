import torch
import torch.nn as nn
import torch.nn.functional as F


class MMOE(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        num_experts,
        expert_hidden,
        num_tasks,
        tower_hidden,
    ):
        super().__init__()

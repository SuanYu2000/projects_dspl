import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class Dist_mar_match(nn.Module):
    def __init__(self, path, gpu=0):
        super().__init__()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)

        directions = torch.load(path, weights_only=True)['w'].T
        self.register_buffer("directions", F.normalize(directions, p=2, dim=1).to(self.device))

        num_centers = self.directions.size(0)
        self.log_scale = nn.Parameter(torch.zeros(num_centers, device=self.device))
        self.register_buffer("queue", torch.randn(num_centers, 128, device=self.device))
        self.register_buffer("polars", self.directions.clone())
        self.matching_indices = None

    def get_scaled_centers(self):
        return self.polars * torch.exp(self.log_scale).unsqueeze(1)

    def forward(self, features, center=None, metric='l2'):
        features = features.to(self.device)
        if center is None:
            center = self.get_scaled_centers()

        if metric == 'l2':
            f_2 = torch.sum(features.pow(2), dim=1, keepdim=True)
            c_2 = torch.sum(center.pow(2), dim=1, keepdim=True)
            dist = f_2 - 2 * features.matmul(center.t()) + c_2.t()
            return dist / features.shape[1]
        else:
            return features.matmul(center.t())

    def update_queue(self, features, labels):
        features = F.normalize(features.to(self.device), p=2, dim=1)
        labels = labels.to(self.device)

        for cls in labels.unique():
            cls_feat = features[labels == cls]
            self.queue[cls] = 0.9 * self.queue[cls] + 0.1 * cls_feat.mean(dim=0)
        self.queue = F.normalize(self.queue, p=2, dim=1)


    @torch.no_grad()
    def update_matching(self):
        
        # print("queue device:", self.queue.device)
        # print("polars device:", self.polars.device)

        score = torch.mm(self.queue, self.directions.t()) 

       
        row_ind, col_ind = linear_sum_assignment(-score.cpu().detach().numpy())
        self.matching_indices = torch.tensor(col_ind, dtype=torch.long, device=self.device)

       
        old_polars = self.polars.clone()
        for i, j in enumerate(self.matching_indices):
            self.polars[:, i] = old_polars[:, j]

        print(f"[Matching Update] Indices: {self.matching_indices}...")

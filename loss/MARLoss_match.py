import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.Dist_mar_match import Dist_mar_match


class MARLoss_match(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(MARLoss_match, self).__init__()
        self.use_gpu = options['use_gpu'] # 没啥用，后期删
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist_mar_match(options['centroid_path'], gpu=0)

        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, y, labels=None):

        # 动态获取最新中心（方向 × 模长）
        centers = self.Dist.get_scaled_centers()

        # 计算两个距离分支
        dist_dot_p = self.Dist(x, center=centers, metric='dot')
        dist_l2_p = self.Dist(x, center=centers, metric='l2')

        # 差值作为 logits
        logits = dist_l2_p - dist_dot_p

        if labels is None:
            return logits, 0

        self.Dist.update_queue(x, labels)

        # 交叉熵 loss
        loss = F.cross_entropy(logits / self.temp, labels)

        # 正则项：距离当前样本到其中心的欧式距离
        center_batch = centers[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones_like(_dis_known).to(x.device)
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r
        return logits, loss

    def fake_loss(self, x):
        centers = self.Dist.get_scaled_centers()
        logits = self.Dist(x, center=centers)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        return loss

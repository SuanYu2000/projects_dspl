import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.Dist_mar_match import Dist_mar_match


class MARLoss_match_angle(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(MARLoss_match_angle, self).__init__()
        self.use_gpu = options['use_gpu'] 
        self.weight_pl = float(options['weight_pl'])
        self.weight_angle = float(options.get('weight_angle', 0.1)) 
        self.temp = options['temp']
        self.Dist = Dist_mar_match(options['centroid_path'], gpu=0)

        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)


    def forward(self, x, y, labels=None):

       
        centers = self.Dist.get_scaled_centers()

    
        dist_dot_p = self.Dist(x, center=centers, metric='dot')
        dist_l2_p = self.Dist(x, center=centers, metric='l2')

     
        logits = dist_l2_p - dist_dot_p

        if labels is None:
            return logits, 0

        self.Dist.update_queue(x, labels)

   
        loss = F.cross_entropy(logits / self.temp, labels)

       
        center_batch = centers[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones_like(_dis_known).to(x.device)
        loss_r = self.margin_loss(self.radius, _dis_known, target)

      
        cos_sim = F.cosine_similarity(x, center_batch, dim=1)
        angle_loss = 1.0 - cos_sim.mean()

       
        loss = loss + self.weight_pl * loss_r + self.weight_angle * angle_loss
        return logits, loss

    def fake_loss(self, x):
        centers = self.Dist.get_scaled_centers()
        logits = self.Dist(x, center=centers)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()
        return loss

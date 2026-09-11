import math
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = sys.float_info.epsilon


def reduce_contrastive_batch_losses(
        loss_terms, reduction='sum', weights=None):
    """Aggregate scalar mini-batch losses with an explicit protocol.

    ``InstanceLoss`` already returns the mean loss within one mini-batch.
    Summing those values therefore makes the effective instance weight grow
    with the number of mini-batches. ``mean`` uses batch sample counts when
    supplied, so a smaller final batch cannot receive the same weight as a
    full batch. ``sum`` is retained for exact legacy reproduction.
    """
    if reduction not in ('sum', 'mean'):
        raise ValueError(
            "instance_batch_reduction must be 'sum' or 'mean'"
        )
    terms = list(loss_terms)
    if not terms:
        raise ValueError('at least one instance-contrastive batch is required')
    stacked = torch.stack(terms)
    if reduction == 'mean':
        if weights is None:
            return stacked.mean()
        weights = list(weights)
        if len(weights) != len(terms):
            raise ValueError('weights must match the number of loss terms')
        weight_tensor = stacked.new_tensor(weights)
        if torch.any(weight_tensor <= 0):
            raise ValueError('batch weights must be positive')
        return (stacked * weight_tensor).sum() / weight_tensor.sum()
    return stacked.sum()


class InstanceLoss(nn.Module):
    """Instance-level contrastive loss"""
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.register_buffer(
            'mask', self.mask_correlated_samples(batch_size).to(device)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        if z_i.shape != z_j.shape or z_i.ndim != 2:
            raise ValueError('paired instance embeddings must have equal 2-D shapes')
        batch_size = int(z_i.shape[0])
        if batch_size < 2:
            raise ValueError('InstanceLoss requires at least two paired samples')
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size == self.batch_size:
            mask = self.mask
        else:
            mask = self.mask_correlated_samples(batch_size).to(sim.device)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    """Cluster-level contrastive loss"""
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.register_buffer(
            'mask', self.mask_correlated_clusters(class_num).to(device)
        )
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + alpha * ne_loss


def JS_divergence(p, q, ):
    """Jensen-Shannon divergence between two distributions"""
    mean = 0.5 * (p + q)
    mean = torch.where(mean < EPS, torch.tensor([EPS], device=mean.device), mean)
    js_div = 0.5 * (F.kl_div(mean.log(), p, reduction='batchmean') + F.kl_div(mean.log(), q, reduction='batchmean'))
    return js_div


def SimCLRLoss(ps, temp=0.1, large_num=1e9):
    """SimCLR contrastive loss"""
    n = ps.size(0) // 2
    h1, h2 = ps[:n], ps[n:]
    h2 = torch.nn.functional.normalize(h2, p=2, dim=1)
    h1 = torch.nn.functional.normalize(h1, p=2, dim=1)

    labels = torch.arange(0, n, device=ps.device, dtype=torch.long)
    masks = torch.eye(n, device=ps.device)

    logits_aa = ((h1 @ h1.t()) / temp) - masks * large_num
    logits_bb = ((h2 @ h2.t()) / temp) - masks * large_num

    logits_ab = (h1 @ h2.t()) / temp
    logits_ba = (h2 @ h1.t()) / temp

    loss_a = torch.nn.functional.cross_entropy(torch.cat((logits_ab, logits_aa), dim=1), labels)
    loss_b = torch.nn.functional.cross_entropy(torch.cat((logits_ba, logits_bb), dim=1), labels)
    loss = (loss_a + loss_b)
    return loss


def SimSiamLoss(h, hs, p, ps):
    """h contains projected features; p contains predicted features"""
    loss1 = - F.cosine_similarity(p, hs.detach(), dim=-1).mean()
    loss2 = - F.cosine_similarity(ps, h.detach(), dim=-1).mean()
    return 0.5 * (loss1 + loss2)


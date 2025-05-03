import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1Criterion(nn.Module):
    """
    Example: Region Query Loss Assignment  (Q = 6)

        [ GT ]   [ GT ]   [ Non-GT ]   [ Non-GT ]   [ Pad ]   [ Pad ]
          │        │         │            │            │         │
          │        │         │            │            │         └─ excluded (has -inf in pred)
          │        │         │            │            └────────────
          │        │         │            │
          │        │         │            └── ③ negative magnitude loss (||v|| ↓)
          │        │         │
          │        │         └─────────────── ③ negative magnitude loss (||v|| ↓)
          │        │
          │        └───────────────────────── ① cosine loss
          │                                  ② positive magnitude loss (||v|| ↑)
          └───────────────────────────────── ① cosine loss
                                             ② positive magnitude loss (||v|| ↑)

    Assumptions:
      - GT queries are always at the front of region slots.
      - Padding entries are at the end and contain -inf.

    Loss Summary:
      ① Cosine loss           → GT only        → alignment (1 - cos_sim)
      ② Positive norm loss    → GT only        → encourage high norm (confident)
      ③ Negative norm loss    → Non-GT only    → suppress norm (background)
      Padding                 → ignored         → filtered using isinf
    """

    def __init__(self, 
                 align_loss_weight=1.0, 
                 pos_magnitude_weight=0.05, 
                 neg_magnitude_weight=0.1):
        super().__init__()
        self.align_loss_weight = align_loss_weight
        self.pos_magnitude_weight = pos_magnitude_weight
        self.neg_magnitude_weight = neg_magnitude_weight

    def forward(self, outputs, targets=None):
        """
        outputs: dict with
            'kformer_output': (B, Q, D) — region embeddings
            'targets': (B, Q, D) — target embeddings (0 for negatives)
        """

        pred = outputs['kformer_output']  # [B, Q, D]
        target = outputs['targets']       # [B, Q, D]
        device = pred.device

        # 1. gt mask: positive sample (target vector is non-zero)
        gt_mask = (target.norm(dim=-1) > 0)      # [B, Q]
        non_gt_mask = ~gt_mask                   # [B, Q]

        # 2. cosine similarity loss (for positives only)
        if gt_mask.any():
            pred_norm = F.normalize(pred[gt_mask], dim=-1)       # [GT(across all batch), Q]
            target_norm = F.normalize(target[gt_mask].detach(), dim=-1)   # [GT(across all batch), Q]
            align_loss = 1.0 - (pred_norm * target_norm).sum(-1).mean()
        else:
            align_loss = torch.tensor(0.0, device=device)

        # 3. magnitude loss: GT-positive should have large norm
        if gt_mask.any():
            pos_magnitude_loss = -pred[gt_mask].norm(dim=-1).mean()  # maximize norm
        else:
            pos_magnitude_loss = torch.tensor(0.0, device=device)

        # 4. magnitude loss: GT-negative should have small norm
        # (Only for valid entries — not -inf padded)
        valid_mask = ~torch.isinf(pred).any(dim=-1)
        valid_neg_mask = non_gt_mask & valid_mask

        if valid_neg_mask.any():
            neg_magnitude_loss = pred[valid_neg_mask].norm(dim=-1).mean()  # minimize norm
        else:
            neg_magnitude_loss = torch.tensor(0.0, device=device)

        # 5. Total loss
        align_loss = align_loss * self.align_loss_weight
        pos_magnitude_loss = pos_magnitude_loss * self.pos_magnitude_weight
        neg_magnitude_loss = neg_magnitude_loss * self.neg_magnitude_weight


        return {
            'loss_align': align_loss,
            'loss_pos_magnitude': pos_magnitude_loss,
            'loss_neg_magnitude': neg_magnitude_loss
        }

def info_nce_multi_positive(query, keys, labels, temperature=0.07):
    """
    query: [N, D]  — region features
    keys:  [M, D]  — target embeddings
    labels: [N]    — int class ids of each query
    """

    # Normalize
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)

    # Cosine similarity matrix: [N, M]
    sim_matrix = torch.matmul(query, keys.T) / temperature

    # Same-class mask: [N, M]
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

    # numerator: sum of similarities to all same-class vectors
    numerator = (sim_matrix * label_mask).exp().sum(dim=1)

    # denominator: all similarities
    denominator = sim_matrix.exp().sum(dim=1)

    loss = -torch.log(numerator / (denominator + 1e-6))
    return loss.mean()
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
                 align_weight=1,
                 contrast_weight=1,
                 pos_mag_weight=1,
                 neg_mag_weight=1
        ):
        super().__init__()
        self.align_weight = align_weight
        self.contrast_weight = contrast_weight
        self.pos_mag_weight = pos_mag_weight
        self.neg_mag_weight = neg_mag_weight

    def forward(self, outputs):
        """
        outputs: dict with
            'kformer_output': (B, Q, D) — region embeddings
            'targets': (B, Q, D) — target embeddings (0 for negatives)
            'gt_labels': List of (Q,) int class ids for each query in the batch. -1 for non-GT
        """
        pred = outputs['kformer_output']  # [B, Q, D]
        target = outputs['targets']       # [B, Q, D]
        gt_labels = outputs['gt_labels']  # [B, Q]
        B, Q, D = pred.shape
        device = pred.device

        # Mask
        gt_label_tensor = torch.full((B, Q), fill_value=-1, dtype=torch.long, device=device)
        for i, labels in enumerate(gt_labels):
            trunc_labels = labels[:Q]
            gt_label_tensor[i, :trunc_labels.shape[0]] = trunc_labels
        gt_mask = gt_label_tensor > -1
        valid_mask = ~torch.isinf(pred).any(dim=-1)
        valid_neg_mask = (~gt_mask) & valid_mask
        label_flat = gt_label_tensor[gt_mask]


        if gt_mask.any():
            pred_gt = pred[gt_mask]  # [N=(All GT across batch), D]
            target_gt = target[gt_mask].detach()

            # 1. Alignment loss (BMSE to BERT target)
            align_loss = info_nce_multi_positive(pred_gt, target_gt, label_flat, self=False)

            # 2. Contrastive loss (between preds)
            # contrast_loss = info_nce_multi_positive(pred_gt, pred_gt, label_flat, self=True)

            # 3. magnitude loss: GT-positive should have large norm
            pos_magnitude_loss = -pred_gt.norm(dim=-1).mean()  # maximize norm

            if valid_neg_mask.any():
                neg_magnitude_loss = pred[valid_neg_mask].norm(dim=-1).mean()  # minimize norm
            else:
                neg_magnitude_loss = torch.tensor(0.0, device=device)
        else:
            align_loss = torch.tensor(0.0, device=device)
            contrast_loss = torch.tensor(0.0, device=device)
            pos_magnitude_loss = torch.tensor(0.0, device=device)
            neg_magnitude_loss = torch.tensor(0.0, device=device)


        # 5. Total loss
        align_loss = align_loss * self.align_weight
        # contrast_loss = contrast_loss * self.contrast_weight
        pos_magnitude_loss = pos_magnitude_loss * self.pos_mag_weight
        neg_magnitude_loss = neg_magnitude_loss * self.neg_mag_weight


        return {
            'align_loss': align_loss,
            # 'contrast_loss': contrast_loss,
            'pos_magnitude_loss': pos_magnitude_loss,
            'neg_magnitude_loss': neg_magnitude_loss,
        }

def info_nce_multi_positive(query, key, labels, self=False, temperature=0.07):
    """
    query: [N, D]  — region features
    keys:  [N, D]  — target embeddings
    labels: [N]    — int class ids of each query
    """

    # Normalize
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)

    # Cosine similarity matrix: [N, N]
    sim_matrix = torch.matmul(query, key.T) / temperature

    # Same-class mask: [N, N]
    label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

    if self:
        self_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        label_mask = (label_mask & self_mask)
    label_mask = label_mask.float()
    if label_mask.sum().item() == 0:
        return torch.tensor(0.0, device=labels.device)

    sim_exp = sim_matrix.exp()

    # numerator: sum of similarities to all same-class vectors
    numerator = (sim_exp * label_mask).sum(dim=1)

    # denominator: all similarities
    denominator = sim_exp.sum(dim=1)

    loss = -torch.log((numerator + 1e-6) / (denominator + 1e-6))
    return loss.mean()
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdamERCTCLoss(nn.Module):
    def __init__(self, blank=0, zero_infinity=False, beta=0.1, epsilon=1e-8):
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        # self.beta = nn.Parameter(torch.tensor(beta))
        self.beta = nn.Parameter(torch.log(torch.exp(torch.tensor(beta)) - 1.0)) # inverse softplus init
        self.epsilon = epsilon

    def compute_entropy(self, log_probs):
        probs = log_probs.exp().clamp(min=self.epsilon)

        # Re-normalizing so that probabilities sum to 1
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Recompute log_probs for numerical consistency
        safe_log_probs = probs.clamp(min=self.epsilon).log()

        # Compute entropy per time-step and average
        entropy = -(probs * safe_log_probs).sum(dim=-1)
        mean_entropy = entropy.mean() # scalar

        return mean_entropy

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # DEBUGGING
        # print("log_probs:", log_probs.shape)
        # print("targets:", targets.shape)
        # print("input_lengths:", input_lengths)
        # print("target_lengths:", target_lengths)
        # print(log_probs) # ok they are negative values
        
        # 1. Standard CTC Loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Entropy regularization
        entropy = self.compute_entropy(log_probs)
        
        # 3. Combined AdaMER-CTC Loss
        # beta = torch.clamp(self.beta, min=0.0) # ensure nonnegative
        beta = F.softplus(self.beta) # should always be > 0
        loss = ctc_loss + beta * entropy

        # print("CTC:", ctc_loss.item())
        # print("Entropy:", entropy.item())
        # print("Combined loss:", loss.item())
        
        return loss

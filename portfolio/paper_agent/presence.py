import torch
from transformers import LogitsProcessor
class GeneratedPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, input_ids[:, self.prompt_length:], True)
        return scores - self.penalty * mask.to(scores.dtype)

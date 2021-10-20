import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LuongAttention', 'BahdanauAttention']


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super().__init__()

        if method not in ['dot', 'general', 'concat']:
            msg = 'method should be dot, general or concat.'
            raise ValueError(msg)
        else:
            self.method = method

        self.hidden_size = hidden_size

        if method == 'general':
            self.attention = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attention = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        self.energy = {
            'dot': self.dot_score,
            'general': self.general_score,
            'concat': self.concat_score
        }
        self.concat_fc = nn.Linear(hidden_size * 2, hidden_size)

    def dot_score(self, hidden, hidden_sequence):
        return torch.sum(hidden * hidden_sequence, dim=2)

    def general_score(self, hidden, hidden_sequence):
        energy = self.attention(hidden_sequence)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, hidden_sequence):
        concat_hidden = torch.cat(
            (hidden.expand(hidden_sequence.size(0), -1, -1), hidden_sequence),
            dim=2)
        energy = torch.tanh(self.attention(concat_hidden))
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, hidden_sequence, input_masks):
        energy = self.energy[self.method](hidden, hidden_sequence)
        energy.masked_fill_(input_masks.logical_not(),
                            value=torch.tensor(-1e10,
                                               dtype=torch.float)).permute(
                                                   1, 0)
        attention_weight = F.softmax(energy,
                                     dim=1).unsqueeze(dim=1).permute(2, 1, 0)
        context = attention_weight.bmm(hidden_sequence.permute(1, 0,
                                                               2)).permute(
                                                                   1, 0, 2)
        return context


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, hidden_seqence, input_masks):
        hidden = hidden.expand(hidden_seqence.size(0), -1, -1)
        energy = self.w1(hidden_seqence) + self.w2(hidden)
        energy = torch.sum(self.v * torch.tanh(energy), dim=2)
        energy = energy.masked_fill_(
            input_masks.logical_not(),
            value=torch.tensor(-1e10, dtype=torch.float)).permute(1, 0)
        attention_weight = F.softmax(energy, dim=1).unsqueeze(1)
        context = attention_weight.bmm(hidden_seqence.permute(1, 0,
                                                              2)).permute(
                                                                  1, 0, 2)
        return context

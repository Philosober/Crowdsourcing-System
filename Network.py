import torch.nn as nn
import torch
from torch.nn import functional as F

class DQN_Net(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=4):
        super(DQN_Net, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

        torch.nn.init.kaiming_normal_(self.fc1[0].weight)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, input):
        # input: B x T x input_size
        input = input.transpose(0, 1)   # T x B x input_size
        input = self.fc1(input)         # T x B x hidden_size
        output, _ = self.attention(input, input, input)   # T x B x hidden_size
        output = output + input   # resnet
        output = self.fc(output)     # T x B x 1
        output = output.transpose(0, 1)   # B x T x 1

        return output

class REINFORCE_Net(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=4):
        super(REINFORCE_Net, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(normalized_shape=hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

        torch.nn.init.kaiming_normal_(self.fc1[0].weight)
        torch.nn.init.kaiming_normal_(self.fc.weight)

        self.saved_log_probs = []
        self.rewards = []
        self.gamma = 0.995

    def forward(self, input):
        # input: B x T x input_size
        input = input.transpose(0, 1)   # T x B x input_size
        input = self.fc1(input)         # T x B x hidden_size
        output, _ = self.attention(input, input, input)   # T x B x hidden_size
        output = output + input   # resnet
        output = self.fc(output)     # T x B x 1
        output = output.transpose(0, 1)   # B x T x 1
        output = F.softmax(output, dim=1)   # B x T x 1      概率化

        return output
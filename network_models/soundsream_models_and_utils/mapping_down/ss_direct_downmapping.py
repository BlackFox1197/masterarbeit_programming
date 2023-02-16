from torch import nn


class SS_Direct_Downmapping_Model(nn.Module):
    def __init__(self,  output,  start_dim, dropout = 0.1):
        super(SS_Direct_Downmapping_Model, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(start_dim, output)

    def forward(self, x):

        x = x.flatten(1)
        x = self.liear(x)
        return x

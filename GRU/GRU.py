import torch
from torch import nn, optim

class GRUcell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUcell, self).__init__()
        self.hidden_size = hidden_size
        # update gate
        self.ux = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)
        # reset gate
        self.rx = nn.Linear(input_size, hidden_size)
        self.rh = nn.Linear(hidden_size, hidden_size)
        # candidate
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        u = torch.sigmoid(self.ux(x) + self.uh(h_prev))
        r = torch.sigmoid(self.rx(x) + self.rh(h_prev))
        c_tilde = torch.tanh(self.W_h(x) + torch.mul(r, self.U_h(h_prev)))
        h = torch.mul(1-u, h_prev) + torch.mul(u, c_tilde)
        return h

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = GRUcell(input_size, hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device) 
        for t in range(x.size(1)):
            hidden = self.cell(x[:, t, :], hidden)
        output = self.fc(hidden)
        return output

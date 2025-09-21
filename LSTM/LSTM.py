import torch
from torch import nn


class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
        # Input gate
    self.W_xi = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
    self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
    self.b_i  = nn.Parameter(torch.zeros(hidden_size))

    # Forget gate
    self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
    self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
    self.b_f  = nn.Parameter(torch.zeros(hidden_size))

    # Output gate
    self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
    self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
    self.b_o  = nn.Parameter(torch.zeros(hidden_size))

    # Candidate cell
    self.W_xc = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
    self.W_hc = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
    self.b_c  = nn.Parameter(torch.zeros(hidden_size))

  def forward(self,x, h_prev, c_prev):
    i = torch.sigmoid(x @ self.W_xi + h_prev @ self.W_hi + self.b_i)
    f = torch.sigmoid(x @ self.W_xf + h_prev @ self.W_hf + self.b_f)
    o = torch.sigmoid(x @ self.W_xo + h_prev @ self.W_ho + self.b_o)
    c_tilde = torch.tanh(x @ self.W_xc + h_prev @ self.W_hc + self.b_c)

    # New cell state and hidden state
    c = f * c_prev + i * c_tilde
    h = o * torch.tanh(c)
    return h, c


class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.cell = LSTMCell(input_size, hidden_size)
    self.fc = nn.Linear(hidden_size, output_size)


  def forward(self, x):
    batch_size, seq_len, _ = x.size()
    h_prev = torch.zeros(batch_size, self.hidden_size,device=device)
    c_prev = torch.zeros(batch_size, self.hidden_size,device=device)
    for t in range(seq_len):
      x_t = x[:, t, :]
      h_t, c_t = self.cell(x_t, h_prev, c_prev)

    out = self.fc(h_t)
    return out

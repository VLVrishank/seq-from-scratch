class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        h = torch.zeros(self.hidden_size, device=self.Wxh.device)
        for x in inputs:
            h = torch.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        y = self.Why @ h + self.by
        return y, h

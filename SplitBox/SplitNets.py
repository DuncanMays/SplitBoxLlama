import torch

class BaseSkipNet(torch.nn.Module):
	
	def __init__(self):
		super().__init__()

		self.ffn = torch.nn.Sequential(
			torch.nn.Linear(20, 100),
			torch.nn.Linear(100, 20),
		)

class OneOneNet(BaseSkipNet):
	
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return self.ffn(x)

class OneTwoNet(BaseSkipNet):
	
	def __init__(self):
		super().__init__()

	def forward(self, x):
		y = self.ffn(x)

		skip = torch.rand_like(y)
		skip.requires_grad = True
		skip.retain_grad()

		return y, skip

class TwoTwoNet(BaseSkipNet):
	
	def __init__(self):
		super().__init__()

	def forward(self, x, skip):
		return self.ffn(x + skip), skip

class TwoOneNet(BaseSkipNet):
	
	def __init__(self):
		super().__init__()

	def forward(self, x, skip):
		return self.ffn(x + skip)
import torch

from llama_blocks import LlamaBlock
from config import MASTER_CONFIG

def main():
	
	net = LlamaBlock(MASTER_CONFIG)

	x = torch.randn([32, 16, MASTER_CONFIG['d_model']], requires_grad=True)
	y = None

	with torch.enable_grad():
		y = net(x)

	loss = torch.sum(torch.abs(y))

	loss.backward()

	print(x.shape)
	print(y.shape)

	# print(x.grad)
	print(x.grad.shape)

if (__name__ == "__main__"): main()
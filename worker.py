# from llama_blocks import llama_blocks, llama_optimizer, scheduler

from sys import argv as args

import axon
import torch

device = "cpu"

def get_arg(default_arg, arg_tag):
    if arg_tag in args:
        index = args.index(arg_tag)
        return args[index + 1]

    else:
        return default_arg

class FnService():

    def __init__(self, net):
        self.net = net
        self.saved_tensors = {}

    def apply(self, ctx_id, x):

        with torch.enable_grad():
            x = x.to(device)
            y = self.net(x)

        self.saved_tensors[ctx_id] = (x, y)

        return y.clone().to('cpu')

    def apply_gradients(self, ctx_id, g):
        g = g.to(device)
        (x, y) = self.saved_tensors[ctx_id]
        del self.saved_tensors[ctx_id]
        y.backward(g)
        x = x.to('cpu')
        return x.grad

port = get_arg(8001, "-p")
tl = axon.HTTP_transport.worker(port=port)

# axon.worker.service(FnService(llama_blocks), 'service_handle', tl=tl)
# axon.worker.service(llama_optimizer, 'optimizer', tl=tl)
# axon.worker.service(scheduler, 'scheduler', tl=tl)

class NeuralBlock():

    def __init__(self, name):
        self.name = name
        self.cache = {}

    def forward(self, msg, call_id):
        self.cache[call_id] = f"{self.name} | {msg}"

    def get_activations(self, call_id):
        return self.cache[call_id]

axon.worker.service(NeuralBlock(get_arg("default", "-n")), 'block', tl=tl)

print(f'Serving on port {port}!')
axon.worker.init()
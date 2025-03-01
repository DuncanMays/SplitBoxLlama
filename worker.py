import torch
import axon

from torch import nn
from torch.nn import functional as F

from config import MASTER_CONFIG, get_arg
from llama_blocks import llama_blocks, scheduler, llama_optimizer

device = "cpu"

class GradientManager():

    def __init__(self, net):
        self.net = net
        self.saved_tensors = {}

    def apply(self, ctx_id, x, save_tensors=False):
        print(f'apply: {ctx_id}')

        with torch.enable_grad():
            y = self.net(x)

        if save_tensors:
            self.saved_tensors[ctx_id] = (x, y)

        return y.clone()

    def apply_gradients(self, ctx_id, g, clear_cache=False):
        print(f'apply_gradients: {ctx_id}')
        
        (x, y) = self.saved_tensors[ctx_id]
        y.backward(g)

        if clear_cache:
            del self.saved_tensors[ctx_id]

        return x.grad

    def clear_cache(self):
        self.saved_tensors = {}

class NeuralBlock():

    def __init__(self, gradman, device=device):
        self.gradman = gradman
        self.device = device

    def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):

        if (x == None):
            # if get input from URL
            remote_block = axon.client.get_stub(URL, stub_type=axon.stubs.SyncStub)
            x = remote_block.get_outputs(call_id, clear_cache=clear_remote_cache)

        x = x.to(self.device)
        y = None

        if (direction =='forward'):
            y = self.gradman.apply(call_id, x, save_tensors=save_tensors)
        
        elif (direction =='backward'):
            y = self.gradman.apply_gradients(call_id, x, clear_cache=clear_local_cache)
        
        else:
            raise BaseException(f'Invalid Direction: {direction}')

        if return_outputs:
            return y.to('cpu')
            
    def get_outputs(self, call_id, clear_cache=False):
        (x, y) = self.gradman.saved_tensors[call_id]
        
        if clear_cache:
            del self.gradman.saved_tensors[call_id]
        
        return y

port = get_arg(8001, "-p")
tl = axon.HTTP_transport.worker(port=port)

gm = GradientManager(llama_blocks)
nb = NeuralBlock(gm)

axon.worker.service(nb, 'neural_block', tl=tl, depth=1)
axon.worker.service(llama_optimizer, 'optimizer', tl=tl, depth=1)
axon.worker.service(scheduler, 'scheduler', tl=tl, depth=1)

print(f'Serving {MASTER_CONFIG['n_layers']} blocks on port {port}!')
axon.worker.init(tl=tl)
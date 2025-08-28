import torch
import axon

from torch import nn
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor

from config import MASTER_CONFIG, get_arg
from llama_blocks import llama_blocks, scheduler, llama_optimizer

device = "cpu"

stub_cache = {}
def get_stub(url):
    if url in stub_cache:
        return stub_cache[url]

    else:
        stub = axon.client.get_stub(url, stub_type=axon.stubs.SyncStub)
        stub_cache[url] = stub
        return stub

class NeuralBlock():

    def __init__(self, net, device=device):
        self.device = device
        self.net = net
        self.saved_inputs = {}
        self.saved_outputs = {}
        self.saved_input_grads = {}
        self.saved_output_grads = {}

    def forward(self, activation_id, clear_cache=False):

        if activation_id not in self.saved_inputs: raise BaseException(f"Input activations not found with ID: {activation_id}")

        x = self.saved_inputs[activation_id]

        with torch.enable_grad():
            y = self.net(x)

        self.saved_outputs[activation_id] = y

        if clear_cache:
            del self.saved_inputs[activation_id]

    def backward(self, activation_id, clear_cache=False):

        if activation_id not in self.saved_inputs: raise BaseException(f"Input activations not found with ID: {activation_id}")
        if activation_id not in self.saved_outputs: raise BaseException(f"Output activations not found with ID: {activation_id}")
        if activation_id not in self.saved_output_grads: raise BaseException(f"Output gradients not found with ID: {activation_id}")
        
        x = self.saved_inputs[activation_id]
        y = self.saved_outputs[activation_id]
        g = self.saved_output_grads[activation_id]

        y.backward(g)

        self.saved_input_grads[activation_id] = x.grad

        if clear_cache:
            del self.saved_inputs[activation_id]
            del self.saved_outputs[activation_id]
            del self.saved_output_grads[activation_id]

    def get_activations(self, activation_id, clear_cache=False):

        if activation_id not in self.saved_outputs: raise BaseException(f"Output activations not found with ID: {activation_id}")

        y = self.saved_outputs[activation_id]
        
        if clear_cache:
            del self.saved_outputs[activation_id]
        
        return y

    def get_gradients(self, activation_id, clear_cache=False):
        if activation_id not in self.saved_input_grads: raise BaseException(f"Input gradients not found with ID: {activation_id}")

        g = self.saved_input_grads[activation_id]
        
        if clear_cache:
            del self.saved_input_grads[activation_id]
        
        return g

    def load_activations(self, activation_id, x):
        self.saved_inputs[activation_id] = x

    def load_gradients(self, activation_id, g):
        self.saved_output_grads[activation_id] = g

    def fetch_activations(self, activation_id, source_URL, clear_cache=False):
        remote_block = get_stub(source_URL)
        x = remote_block.get_activations(activation_id, clear_cache=clear_cache)
        self.saved_inputs[activation_id] = x

    def fetch_gradients(self, activation_id, source_URL, clear_cache=False):
        remote_block = get_stub(source_URL)
        g = remote_block.get_gradients(activation_id, clear_cache=clear_cache)
        self.saved_output_grads[activation_id] = g

    def clear_cache(self, activation_id=None):
        
        if (activation_id == None):
            self.saved_inputs = {}
            self.saved_outputs = {}
            self.saved_input_grads = {}
            self.saved_output_grads = {}

        else:
            if activation_id in self.saved_inputs: del self.saved_inputs[activation_id]
            if activation_id in self.saved_outputs: del self.saved_outputs[activation_id]
            if activation_id in self.saved_input_grads: del self.saved_input_grads[activation_id]
            if activation_id in self.saved_output_grads: del self.saved_output_grads[activation_id]

def main():
    port = get_arg(8001, "-p")
    tl = axon.HTTP_transport.worker(port=port)
    tpe = ThreadPoolExecutor(10)

    nb = NeuralBlock(llama_blocks)

    axon.worker.service(nb, 'neural_block', tl=tl, depth=1, executor=tpe)
    axon.worker.service(llama_optimizer, 'optimizer', tl=tl, depth=1, executor=tpe)
    axon.worker.service(scheduler, 'scheduler', tl=tl, depth=1, executor=tpe)

    print(f'Serving {MASTER_CONFIG['n_layers']} blocks on port {port}!')
    axon.worker.init(tl=tl)

if (__name__ == "__main__"):
    main()
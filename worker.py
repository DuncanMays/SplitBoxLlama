import torch
import axon
import cloudpickle

from torch import nn
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor

from config import MASTER_CONFIG, get_arg
from llama_blocks import llama_blocks, scheduler, llama_optimizer

device = "cpu"

# this class is concerned with representing the neural network parameters and optimizer in a way that's easy to move between workers
class NeuralBlock():

    def __init__(self, block_fn):
        self.block_fn = block_fn
        self.net = self.block_fn()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            betas=(.9, .95),
            weight_decay=.1,
            eps=1e-9,
            lr=1e-3
        )

    def __call__(self, x):
        return self.net(x)

    def get_state(self):
        
        state = {}

        state['fn_str'] = cloudpickle.dumps(self.block_fn)
        state['net_state'] = self.net.state_dict()
        state['optimizer_state'] = self.optimizer.state_dict()

        return state

    def set_state(self, state):
        
        self.block_fn = cloudpickle.loads(state['fn_str'])
        self.net = self.block_fn()
        self.net.load_state_dict(state['net_state'])

        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.optimizer.load_state_dict(state['optimizer_state'])

    @classmethod
    def from_state(self, state):

        block = NeuralBlock(lambda : torch.nn.Linear(1, 1))
        block.set_state(state)

        return block

# this class is concerned with the logic of maintaining a stack of NeuralBlocks, trading them with other workers, etc.
class BlockStack():

    def __init__(self):
        self.blocks = []
        self.block_fn = None

    def __call__(self, x):
        
        for block in self.blocks:
            x = block(x)

        return x

    def push_block(self, fn_str, save_fn=True, back=False):
        
        block_fn = self.block_fn

        if (fn_str != None): block_fn = cloudpickle.loads(fn_str)

        if (block_fn == None): raise BaseException("Block function not set")

        if save_fn: self.block_fn = block_fn

        new_block = block_fn()

        if back:
            self.blocks.append(new_block)

        else:
            self.blocks = [new_block] + self.blocks

    def pop_block(self, back=False):

        if back:
            block = self.blocks.pop()

        else:
            block = self.blocks.pop(0)

        return block.parameters()

    def set_parameters(self, index):
        pass

    def get_parameters(self, index):
        return self.blocks[index].paramters()

stub_cache = {}
def get_stub(url):
    if url in stub_cache:
        return stub_cache[url]

    else:
        stub = axon.client.get_stub(url, stub_type=axon.stubs.SyncStub)
        stub_cache[url] = stub
        return stub

# this class is concerned with retrieving and storing activations and gratients to support backpropagation
class Worker():

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
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

class GradientManager():

    def __init__(self, net):
        self.net = net
        self.saved_tensors = {}

    def apply(self, ctx_id, x):

        with torch.enable_grad():
            y = self.net(x)

        self.saved_tensors[ctx_id] = (x, y)

        return y.clone()

    def apply_gradients(self, ctx_id, g):
        (x, y) = self.saved_tensors[ctx_id]
        del self.saved_tensors[ctx_id]
        y.backward(g)
        return x.grad

class NeuralBlock():

    def __init__(self, gradman, device="cpu"):
        self.gradman = gradman
        self.device = device

    def apply(self, x, URL, call_id, return_outputs=False):
        net = self.gradman.apply
        y = self.run_net(x, URL, call_id, net, return_outputs=return_outputs)
        return y
    
    def apply_gradients(self, g, URL, call_id, return_outputs=False):
        net = self.gradman.apply_gradients
        x_grad = self.run_net(g, URL, call_id, net, return_outputs=return_outputs)
        return x_grad

    def run_net(self, x, URL, call_id, net, return_outputs=False):

        if (x == None):
            stub = axon.client.get_stub(URL, stub_type=axon.stubs.SyncStub)
            x = stub.get_outputs(call_id)

        x = x.to(self.device)
        y = net(call_id, x)

        if return_outputs:
            return y.to('cpu')
            
    def get_outputs(self, call_id):
        (x, y) = self.gradman.saved_tensors[call_id]
        return y

port = get_arg(8001, "-p")
tl = axon.HTTP_transport.worker(port=port)

net = torch.nn.Linear(100, 100)
gm = GradientManager(net)
nb = NeuralBlock(gm)

axon.worker.service(nb, 'block', tl=tl, depth=1)

print(f'Serving on port {port}!')
axon.worker.init(tl=tl)

# axon.worker.service(FnService(llama_blocks), 'service_handle', tl=tl)
# axon.worker.service(llama_optimizer, 'optimizer', tl=tl)
# axon.worker.service(scheduler, 'scheduler', tl=tl)
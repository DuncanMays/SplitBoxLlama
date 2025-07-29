
# from pipeline_parallel.py
class MockNeuralBlock():

    def __init__(self, gradman, device=device):
        pass

    def run_net(self, x, URL, direction, call_id, return_outputs=False, clear_local_cache=False, clear_remote_cache=False, save_tensors=False):
        pass
            
    def get_outputs(self, call_id, clear_cache=False):
        pass


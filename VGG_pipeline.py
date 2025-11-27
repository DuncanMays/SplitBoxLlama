import torch
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10
import json
from tqdm import tqdm
import pickle

import axon
import asyncio
import time
import uuid
import cloudpickle

from SplitBox.benchmark import benchmark
from SplitBox.allocation import allocate, round_with_sum_constraint, delay
from SplitBox.plot_pipeline import metrics_wrapper, plot_timings
from SplitBox.multi_stub import get_multi_stub
from SplitBox.worker import NeuralBlock

from VGG_blocks import VGGBlock_1, VGGBlock_2, VGGBlock_3
from config import MASTER_CONFIG
from pipeline_client import get_training_flow

print('starting')

# torch.cuda.empty_cache()

BATCH_SIZE = 32

# device = 'cuda:0'
device = 'cpu'

def get_random_sample(num_samples, x_data, y_data):
    # gets num_samples random indices of elements in data
    indices = torch.randperm(x_data.shape[0])[0:num_samples]
    # puts the tensors alluded to in indices into an array
    x_slices = [x_data[index] for index in indices]
    y_slices = [y_data[index] for index in indices]
    # concatenates and returns those tensors
    return torch.stack(x_slices), torch.stack(y_slices)

# we assume that both y_hat and y_data are one-hot tensors, representing the class
def get_accuracy(y_hat, y_data):
    # The correctness of a prediction on a sample will be the dot product of the prediction with the ground truth
    # taking the element-wise product and then summing is equivalient to summing the output of every dot product on every sample
    # avg = torch.sum(y_hat*y_data)/y_hat.shape[0]

    avg = torch.dot(y_hat.flatten(), y_data.flatten())/y_hat.shape[0]

    return avg.tolist()


I = torch.eye(10)
def to_one_hot(indices):
    indices = torch.flatten(indices).tolist()
    return torch.stack([I[index] for index in indices])

print('importing data')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('formatting inputs')
x_train = torch.tensor(x_train, dtype=torch.float32)/255.
x_train = x_train.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
x_test = torch.tensor(x_test, dtype=torch.float32)/255.0
x_test = x_test.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)

print('formatting outputs')
y_train = torch.tensor([elem[0] for elem in y_train], dtype=torch.long)
y_test = torch.tensor([elem[0] for elem in y_test], dtype=torch.long)

async def main():
    print('instantiating network')

    url_1 = "192.168.2.19:8001/llama_worker"
    url_2 = "192.168.2.19:8002/llama_worker"
    url_3 = "192.168.2.19:8003/llama_worker"
    # url_3 = "192.168.2.44:8001/llama_worker"

    urls = [url_1, url_2, url_3]
    stubs = [axon.client.get_stub(url) for url in urls]

    global_stub = get_multi_stub(stubs)

    block_stubs = [axon.client.get_stub(url+"/net") for url in urls]
    multi_block_stub = get_multi_stub(block_stubs)

    blocks = [NeuralBlock(VGGBlock_1), NeuralBlock(VGGBlock_2), NeuralBlock(VGGBlock_3)]
    block_states = [block.get_state() for block in blocks]

    # await multi_block_stub.load_blocks(block_states)

    # print('training '+statDir)
    NUM_BATCHES = x_train.shape[0]//BATCH_SIZE

    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []

    criterion = torch.nn.CrossEntropyLoss()
    # mse = torch.nn.MSELoss()
    # criterion = lambda y_hat, y_batch : mse(y_hat, to_one_hot(y_batch).to(device))

    while(True):
        # print("epoch: "+str(epoch))

        for j in tqdm(range(NUM_BATCHES)):
            # store the training losses and accuracies for each batch in this epoch
            # training_losses_epoch = []
            # training_accuracies_epoch = []

            x_batch = x_train[BATCH_SIZE*j: BATCH_SIZE*(j+1)]
            y_batch = y_train[BATCH_SIZE*j: BATCH_SIZE*(j+1)]
            # batch = torch.randn([num_mini_batches, 32, 16, MASTER_CONFIG['d_model']])
            # target = torch.randn([32, 16, MASTER_CONFIG['d_model']])

            flow, losses = get_training_flow(urls, stubs, x_batch, y_batch)

            print('executing training flow')
            await flow.start()
            print(losses)

            print('optimizer step')
            await multi_block_stub.step([{"zero_grad": True} for _ in stubs])

            print('clearing cache')
            await global_stub.clear_cache()

            # # appends data to training stats
            # y_batch_vec = to_one_hot(y_batch).to(device)
            # training_acc = get_accuracy(y_hat, y_batch_vec)

            # training_losses_epoch.append(loss.item())
            # training_accuracies_epoch.append(training_acc)

        # we now average the training losses and accuracy for each batch this epoch
        # training_loss = sum(training_losses_epoch)/len(training_losses_epoch)
        # training_acc = sum(training_accuracies_epoch)/len(training_accuracies_epoch)

        # testing_loss, testing_acc = test()

        # print('estimated training accuracy was: '+str(training_acc)+' and testing accuracy was: '+str(testing_acc))

        # print('saving training state')

        # params_list = list(net.parameters())
        # marshalled = {'params':params_list, 'epoch':epoch+1}
        # byte_strm = pickle.dumps(marshalled)
        # f = open(save_path, 'wb')
        # f.write(byte_strm)
        # f.close()

        # training_losses.append(training_loss)
        # training_accuracies.append(training_acc)
        # testing_losses.append(testing_loss)
        # testing_accuracies.append(testing_acc)

        # epoch += 1


asyncio.run(main())
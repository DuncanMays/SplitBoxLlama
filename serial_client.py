from collections import OrderedDict

# PyTorch for implementing LLM (No GPU)
import torch

# Neural network modules and functions from PyTorch
from torch import nn
from torch.nn import functional as F
# NumPy for numerical operations
import numpy as np
# Matplotlib for plotting Loss etc.
from matplotlib import pyplot as plt
# Time module for tracking execution time
import time
# Pandas for data manipulation and analysis
import pandas as pd
from tqdm import tqdm

from config import MASTER_CONFIG

# Read the content of the dataset
lines = open("tinyshakespeare.txt", 'r').read()

# Create a sorted list of unique characters in the dataset
vocab = sorted(list(set(lines)))
MASTER_CONFIG.update(vocab_size=len(vocab))

MASTER_CONFIG.update(vocab_size=len(vocab))

# Mapping integers to characters (itos)
itos = {i: ch for i, ch in enumerate(vocab)}

# Mapping characters to integers (stoi)
stoi = {ch: i for i, ch in enumerate(vocab)}

# Encode function: Converts a string to a list of integers using the mapping stoi
def encode(s):
    return [stoi[ch] for ch in s]

# Decode function: Converts a list of integers back to a string using the mapping itos
def decode(l):
    return ''.join([itos[i] for i in l])

# Convert the dataset into a torch tensor with specified data type (dtype)
dataset = torch.tensor(encode(lines), dtype=torch.int8)

# Function to get batches for training, validation, or testing
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # Split the dataset into training, validation, and test sets
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    # Determine which split to use
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # Pick random starting points within the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y

@torch.no_grad()  # Don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    # Placeholder for the evaluation results
    out = {}
    
    # Set the model to evaluation mode
    model.eval()
    # print(model.embeddings.weight.device)
    # model.to('cuda:0')

    # Iterate through training and validation splits
    for split in ["train", "val"]:
        # Placeholder for individual losses
        losses = []

        # Generate 10 batches for evaluation
        for _ in range(10):
            # Get input sequences (xb) and target sequences (yb)
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])

            # xb = xb.to('cuda:0')
            # yb = yb.to('cuda:0')
            
            # Perform model inference and calculate the loss
            _, loss = model(xb, yb)
            
            # Append the loss to the list
            losses.append(loss.item())

        # Calculate the mean loss for the split and store it in the output dictionary
        out[split] = np.mean(losses)
    
    # Set the model back to training mode
    model.train()
    model.to('cpu')
    
    return out

# Function to perform training
def train(model, optimizer, remote_optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # Placeholder for storing losses
    losses = []
    
    # Start tracking time
    start_time = time.time()

    # model.to('cuda:0')

    remote_scheduler = None
    if scheduler:
        remote_scheduler = scheduler[1]
        scheduler = scheduler[0]

    # Iterate through epochs
    for epoch in tqdm(range(config['epochs'])):
        # Zero out gradients
        optimizer.zero_grad()
        remote_optimizer.zero_grad()

        # Obtain batches for training
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # xs = xs.to('cuda:0')
        # ys = ys.to('cuda:0')

        # Forward pass through the model to calculate logits and loss
        logits, loss = model(xs, targets=ys)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        remote_optimizer.step()

        # If a learning rate scheduler is provided, adjust the learning rate
        if scheduler:
            scheduler.step()
            remote_scheduler.step()

        # Log progress every specified interval
        if epoch % config['log_interval'] == 0:
            # Calculate batch time
            batch_time = time.time() - start_time
            
            # Evaluate loss on validation set
            x = evaluate_loss(model)
            
            # Store the validation loss
            losses += [x]
            
            # Print progress logs if specified
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
                
            # Reset the timer
            start_time = time.time()

            # Print learning rate if a scheduler is provided
            if scheduler:
                print("lr: ", scheduler.get_lr())

    model.to('cpu')

    # Print the final validation loss
    print("Validation loss: ", losses[-1]['val'])
    
    # Plot the training and validation loss curves
    return pd.DataFrame(losses).plot()

import axon
import uuid

from llama_blocks import SwiGLU

worker_ip = '192.168.2.19'
port = 8001

class Llama(nn.Module):
    def __init__(self, config, llama_blocks):
        super().__init__()
        self.config = config
        # Embedding layer for token representations
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])

        self.llama_blocks = llama_blocks

        # Feedforward network (FFN) for final output
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # Print total number of parameters in the model
        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # Input token indices are passed through the embedding layer
        # self.embeddings.to('cuda:0')
        x = self.embeddings(idx)

        # Process the input through the LlamaBlocks
        # self.llama_blocks.to('cuda:0')
        x = self.llama_blocks.apply(x)
        
        # Pass the processed input through the final FFN for output logits
        # self.ffn.to('cuda:0')
        logits = self.ffn(x)

        # If targets are not provided, return only the logits
        if targets is None:
            return logits
        # If targets are provided, compute and return the cross-entropy loss
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

url_1 = "localhost:8001/neural_block"
url_2 = "localhost:8002/neural_block"
url_3 = "localhost:8003/neural_block"

print('creating stubs')
stub_1 = axon.client.get_stub(url_1, stub_type=axon.stubs.SyncStub)
stub_2 = axon.client.get_stub(url_2, stub_type=axon.stubs.SyncStub)
stub_3 = axon.client.get_stub(url_3, stub_type=axon.stubs.SyncStub)

stub_1.clear_cache()
stub_2.clear_cache()
stub_3.clear_cache()

class FnStub(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, x):

        # if the context already has an ID, that means it's been through a FnStub already
        # this could be a problem for recursive patterns, in that case, the stub's context store will need to be a dict of lists of contexts
        if not hasattr(ctx, 'id'):
            ctx.id = uuid.uuid4()   

        stub_1.load_activations(ctx.id, x)
        stub_1.forward(ctx.id)

        stub_2.fetch_activations(ctx.id, url_1)
        stub_2.forward(ctx.id)

        stub_3.fetch_activations(ctx.id, url_2)
        stub_3.forward(ctx.id)
        x = stub_3.get_activations(ctx.id)

        return x

    @staticmethod
    def backward(ctx, g):

        stub_3.load_gradients(ctx.id, g)
        stub_3.backward(ctx.id, clear_cache=True)

        stub_2.fetch_gradients(ctx.id, url_3, clear_cache=True)
        stub_2.backward(ctx.id, clear_cache=True)

        stub_1.fetch_gradients(ctx.id, url_2, clear_cache=True)
        stub_1.backward(ctx.id, clear_cache=True)
        x = stub_1.get_gradients(ctx.id, clear_cache=True)

        return g

# Create Llama model with Cosine Annealing learning schedule
llama_with_cosine = Llama(MASTER_CONFIG, FnStub)

# Define Adam optimizer with specific hyperparameters
llama_optimizer = torch.optim.Adam(
    llama_with_cosine.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)

# Define Cosine Annealing learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)

remote_scheduler = axon.client.get_stub(f'{worker_ip}:{port}/scheduler', stub_type=axon.stubs.SyncStub)
remote_optimizer = axon.client.get_stub(f'{worker_ip}:{port}/optimizer', stub_type=axon.stubs.SyncStub)

# Train the Llama model with the specified optimizer and scheduler
train(llama_with_cosine, llama_optimizer, remote_optimizer, scheduler=(scheduler, remote_scheduler))

plt.show()
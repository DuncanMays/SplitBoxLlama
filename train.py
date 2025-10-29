import torch
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10
import json
from tqdm import tqdm
import pickle

print('starting')

# torch.cuda.empty_cache()

BATCH_SIZE = 64

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

def test_on_sample(num_samples):
    x_train_sample, y_train_sample = get_random_sample(num_samples, x_train, y_train)
    x_test_sample, y_test_sample = get_random_sample(num_samples, x_test, y_test)
    
    y_train_hat = torch.sigmoid(net(x_train_sample))
    y_test_hat = torch.sigmoid(net(x_test_sample))
    
    y_train_sample_vec = to_one_hot(y_train_sample).to(device)
    y_test_sample_vec = to_one_hot(y_test_sample).to(device)
    
    training_acc = get_accuracy(y_train_hat, y_train_sample_vec)
    testing_acc = get_accuracy(y_test_hat, y_test_sample_vec)

    y_train_sample = y_train_sample.to(device)
    y_test_sample = y_test_sample.to(device)

    training_loss = criterion(y_train_hat, y_train_sample).item()
    testing_loss = criterion(y_test_hat, y_test_sample).item()

    return training_loss, training_acc, testing_loss, testing_acc

def test():
    net.eval()

    NUM_TESTING_BATCHES = x_test.shape[0] // BATCH_SIZE
    for i in range(NUM_TESTING_BATCHES):
        x_test_sample = x_test[i*BATCH_SIZE: (i+1)*BATCH_SIZE].to(device)
        y_test_sample = y_test[i*BATCH_SIZE: (i+1)*BATCH_SIZE].to(device)

        y_test_hat = torch.sigmoid(net(x_test_sample))
        
        y_test_sample_vec = to_one_hot(y_test_sample).to(device)
        
        testing_acc = get_accuracy(y_test_hat, y_test_sample_vec)

        y_test_sample = y_test_sample.to(device)

        testing_loss = criterion(y_test_hat, y_test_sample).item()

    net.train()
    return testing_loss, testing_acc


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

print('instantiating network')
# make sure that this directory actually exists!!!
statDir = 'VGG'
import VGG
net = VGG.VGG_19_skip()

epoch = 0
save_path = './'+statDir+'/VGG_19_skip.save'

# print("loading params")
# f = open(save_path, 'rb')
# byte_strm = f.read()
# f.close()
# marshalled = pickle.loads(byte_strm)
# params_list = marshalled['params']
# epoch = marshalled['epoch']
# current_params = list(net.parameters())
# for i in range(len(params_list)):
#    current_params[i].data = params_list[i].data

net.to(device)

optimizer = torch.optim.Adam(params=net.parameters(), weight_decay=1e-3, lr=0.00001)

print('training '+statDir)
NUM_BATCHES = x_train.shape[0]//BATCH_SIZE

training_losses = []
training_accuracies = []
testing_losses = []
testing_accuracies = []

criterion = torch.nn.CrossEntropyLoss()
# mse = torch.nn.MSELoss()
# criterion = lambda y_hat, y_batch : mse(y_hat, to_one_hot(y_batch).to(device))

while(True):
    print("epoch: "+str(epoch))

    for j in tqdm(range(NUM_BATCHES)):
        # store the training losses and accuracies for each batch in this epoch
        training_losses_epoch = []
        training_accuracies_epoch = []

        # perform gradient update
        optimizer.zero_grad()

        x_batch = x_train[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)
        y_batch = y_train[BATCH_SIZE*j: BATCH_SIZE*(j+1)].to(device)

        y_hat = torch.sigmoid(net(x_batch))
        loss = criterion(y_hat, y_batch)
        
        loss.backward()
        optimizer.step()
        
        # appends data to training stats
        y_batch_vec = to_one_hot(y_batch).to(device)
        training_acc = get_accuracy(y_hat, y_batch_vec)

        training_losses_epoch.append(loss.item())
        training_accuracies_epoch.append(training_acc)

    # we now average the training losses and accuracy for each batch this epoch
    training_loss = sum(training_losses_epoch)/len(training_losses_epoch)
    training_acc = sum(training_accuracies_epoch)/len(training_accuracies_epoch)

    testing_loss, testing_acc = test()

    print('estimated training accuracy was: '+str(training_acc)+' and testing accuracy was: '+str(testing_acc))

    # print('saving training state')

    # params_list = list(net.parameters())
    # marshalled = {'params':params_list, 'epoch':epoch+1}
    # byte_strm = pickle.dumps(marshalled)
    # f = open(save_path, 'wb')
    # f.write(byte_strm)
    # f.close()

    training_losses.append(training_loss)
    training_accuracies.append(training_acc)
    testing_losses.append(testing_loss)
    testing_accuracies.append(testing_acc)

    epoch += 1

# print('saving stats')

# f = open('./'+statDir+'/training_losses', 'w')
# f.write(json.dumps(training_losses))
# f.close()

# f = open('./'+statDir+'/training_accuracies', 'w')
# f.write(json.dumps(testing_accuracies))
# f.close()

# f = open('./'+statDir+'/testing_losses', 'w')
# f.write(json.dumps(testing_losses))
# f.close()

# f = open('./'+statDir+'/testing_accuracies', 'w')
# f.write(json.dumps(testing_accuracies))
# f.close()


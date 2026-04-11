import torch
import torch.nn as nn

from tqdm import tqdm

from ResNetStages import ResNetStage0, ResNetStage1, ResNetStage2
from CIFAR10_data import x_train, y_train, x_test, y_test

print('starting')

torch.cuda.empty_cache()

BATCH_SIZE = 64

device = 'cuda:0'
# device = 'cpu'

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

print('instantiating network')

stage0 = ResNetStage0(num_blocks=18)
stage1 = ResNetStage1(num_blocks=18)
stage2 = ResNetStage2(num_blocks=18, num_classes=10)

net = nn.Sequential(stage0, stage1, stage2)

epoch = 0
net.to(device)

optimizer = torch.optim.Adam(params=net.parameters(), weight_decay=1e-3, lr=0.00001)

print('training '+str(type(net)))
NUM_BATCHES = x_train.shape[0]//BATCH_SIZE

training_losses = []
training_accuracies = []
testing_losses = []
testing_accuracies = []

criterion = torch.nn.CrossEntropyLoss()

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

    testing_loss, testing_acc = test()

    print('estimated training accuracy was: '+str(training_acc)+' and testing accuracy was: '+str(testing_acc))

    epoch += 1

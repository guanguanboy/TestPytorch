
import torch

import torch.nn.functional as F
def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    #### START CODE HERE ####
    return F.one_hot(labels, n_classes)
    #### END CODE HERE ####

labels = [1, 2, 3, 4, 5]
labels = torch.tensor(labels)
n_classes =  10
device = 'cuda'

one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
print(one_hot_labels.shape) #torch.Size([5, 10])
print(one_hot_labels)
image_one_hot_labels = one_hot_labels[:, :, None, None] #torch.Size([5, 10, 1, 1])
print(image_one_hot_labels.shape)

image_one_hot_labels = image_one_hot_labels.repeat(1, 1, 28, 28)
print(image_one_hot_labels.shape) #torch.Size([5, 10, 28, 28])

print(image_one_hot_labels[0][1]) # 全1
print(image_one_hot_labels[0][1].shape) #torch.Size([28, 28])

print(image_one_hot_labels[1][2]) #全1
print(image_one_hot_labels[1][2].shape) #torch.Size([28, 28])

print(image_one_hot_labels[1][3]) #全0

print(image_one_hot_labels[1][3].shape) #torch.Size([28, 28])
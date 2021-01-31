from torch.utils.data import DataLoader
from autoencoder_homework_lihongyi.autoencoder import *
import matplotlib.pyplot as plt

trainX = np.load('trainX.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

print(type(trainX))
print(trainX.shape)
print(trainX.dtype)
print(trainX.__len__())
print(trainX[0].shape)

plt.imshow(trainX[0])
plt.show()

print(trainX_preprocessed.shape)
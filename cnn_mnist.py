# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307, ), (0.3081, ))                           
])

#读取训练集
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                 train=True,
                 download=True,
                 transform=transform)

train_loader = DataLoader(train_dataset,
              shuffle=True,
              batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                train=False,
                download=True,
                transform=transform)

test_loader = DataLoader(test_dataset,
              shuffle=False,
              batch_size=batch_size)

#搭建神经网络
class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5,5))
    self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5,5))
    self.pooling = torch.nn.MaxPool2d(2)
    self.fc = torch.nn.Linear(320, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = F.relu(self.pooling(self.conv1(x)))
    x = F.relu(self.pooling(self.conv2(x)))
    x = x.view(batch_size, -1)
    x = self.fc(x)
    return x

#创建模型
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#训练
def train(epoch):
  running_loss = 0
  for batch_idx,data in enumerate(train_loader,0):
    inputs, target = data
    inputs, target = inputs.cuda(),target.cuda()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if batch_idx%300 ==299:
      print('[%d, %5d] loss: %.3f'%(epoch+1, batch_idx+1, running_loss/300))

#测试
def test():
  correct=0
  total=0
  with torch.no_grad():
    for data in test_loader:
      images, labels = data
      images, labels = images.cuda(),labels.cuda()
      outputs = model(images)
      _,predicted = torch.max(outputs.data, dim=1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  print('Accuracy on test_set: %d %%' % (100*correct/total))

#训练并测试
if __name__ == '__main__':
  for epoch in range(2):
    train(epoch)
    test()

#修改测验图片的尺寸以适配训练集尺寸
from PIL import Image
def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)

import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torchvision
from skimage import io,transform

#识别目标图片数字
if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()    #把模型转为test模式
    #img = cv2.imread("drive/MyDrive/Machine Learning/Pytorch/1.jpg")  #读取要预测的图片

    file_in = "drive/MyDrive/Machine Learning/Pytorch/1.jpg"
    width = 28
    height = 28
    file_out = '23.png'

    

    produceImage(file_in, width, height, file_out)

    img = cv2.imread(file_out)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#图片转为灰度图，因为mnist数据集都是灰度图
    img=np.array(img).astype(np.float32)
    img=np.expand_dims(img,0)
    img=np.expand_dims(img,0)#扩展后，为[1，1，28，28]
    img=torch.from_numpy(img)
    print((img).size())
    img = img.to(device)
    
    output=model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  #用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  #prob是10个分类的概率
    pred = np.argmax(prob) #选出概率最大的一个
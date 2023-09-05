import torch

device = 'cpu'
if torch.cuda.is_available():
    device='cuda'

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,8,3,padding='same')
        self.batchnorm2 = torch.nn.BatchNorm2d(8)
        self.pool3 = torch.nn.MaxPool2d(2,2) # 8,16,16
        self.conv4 = torch.nn.Conv2d(8,16,3,padding=1) # 16,16,16
        self.batchnorm5 = torch.nn.BatchNorm2d(16)
        self.conv6 = torch.nn.Conv2d(16,16,3,padding=1) # 16,16,16
        self.batchnorm7 = torch.nn.BatchNorm2d(16)
        self.pool8 = torch.nn.MaxPool2d(2,2) # 16,8,8
        self.conv9 = torch.nn.Conv2d(16,16,3,padding=1) # 16,8,8
        self.batchnorm10 = torch.nn.BatchNorm2d(16)
        self.pool11 = torch.nn.MaxPool2d(2,2) # 16,4,4
        self.conv12 = torch.nn.Conv2d(16,32,3,padding=1) # 32,4,4
        self.batchnorm13 = torch.nn.BatchNorm2d(32)
        self.conv14 = torch.nn.Conv2d(32,32,3,padding=1) # 32,4,4
        self.batchnorm15 = torch.nn.BatchNorm2d(32)
        self.pool16 = torch.nn.AvgPool2d(4,4) # 32,1,1
        self.fc17 = torch.nn.Linear(32,32) # 16,32
        self.batchnorm18 = torch.nn.BatchNorm1d(32)
        self.fc19 = torch.nn.Linear(32,16) # 16,16
        self.batchnorm20 = torch.nn.BatchNorm1d(16)
        self.fc21 = torch.nn.Linear(16,10)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x.to(device)
        output = self.relu(self.batchnorm2(self.conv1(x)))
        output = self.pool3(output)
        output = self.relu(self.batchnorm5(self.conv4(output)))
        output = self.relu(self.batchnorm7(self.conv6(output)))
        output = self.pool8(output)
        output = self.relu(self.batchnorm10(self.conv9(output)))
        output = self.pool11(output)
        output = self.relu(self.batchnorm13(self.conv12(output)))
        output = self.relu(self.batchnorm15(self.conv14(output)))
        output = self.pool16(output)
        output = torch.reshape(output,(-1,32))
        output = self.relu(self.batchnorm18(self.fc17(output)))
        output = self.relu(self.batchnorm20(self.fc19(output)))
        output = self.fc21(output)
        output = torch.nn.functional.log_softmax(output,dim=1)
        return output
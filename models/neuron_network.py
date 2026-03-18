import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Lớp vào 784 (ảnh 28x28), Lớp ẩn 128
        self.hidden = nn.Linear(784, 128) 
        # Lớp ra 10 (chữ số từ 0-9)
        self.output = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)   # Flatten ảnh thành vector
        x = self.hidden(x)    # Bước 1: Nhân ma trận lớp ẩn
        x = self.relu(x)      # Bước 2: Hàm kích hoạt ReLU
        x = self.output(x)    # Bước 3: Nhân ma trận lớp ra
        return x
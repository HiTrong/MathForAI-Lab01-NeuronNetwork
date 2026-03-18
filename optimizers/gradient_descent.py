import torch
import torch.optim as optim

class MySGD:
    def __init__(self, loss_function, parameters, lr=0.01):
        self.criterion = loss_function
        self.optimizer = optim.SGD(
            params=parameters,
            lr=lr
        )
        self.total_loss = 0
        self.optimize_step = 0
    
    def reset(self):
        self.total_loss = 0
        self.optimize_step = 0

    def optimize(self, outputs, labels):
        self.optimizer.zero_grad()
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.total_loss += loss.item()
        self.optimize_step += 1


        

    

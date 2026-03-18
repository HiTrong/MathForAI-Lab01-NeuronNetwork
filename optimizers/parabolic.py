import copy
import torch

class Parabolic:
    def __init__(self, loss_function, lr=0.01):
        self.lr = lr
        self.criterion = loss_function
        self.total_loss = 0
        self.optimize_step = 0

    def compute_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def reset(self):
        self.total_loss = 0
        self.optimize_step = 0

    def optimize(self, model, images, labels):
        outputs = model(images)
        loss = self.compute_loss(outputs, labels)

        model.zero_grad()
        loss.backward()

        grads = [p.grad.clone() for p in model.parameters()]

        alphas = [0.0, self.lr, self.lr*2]
        losses = []

        for a in alphas:

            temp_model = copy.deepcopy(model)

            with torch.no_grad():
                for p, g in zip(temp_model.parameters(), grads):
                    p -= a * g

            l = self.compute_loss(temp_model(images), labels)
            losses.append(l)

        a1,a2,a3 = alphas
        f1,f2,f3 = losses

        numerator = (a2-a1)**2*(f2-f3) - (a2-a3)**2*(f2-f1)
        denominator = (a2-a1)*(f2-f3) - (a2-a3)*(f2-f1)

        if abs(denominator) < 1e-8:
            alpha_star = self.lr
        else:
            alpha_star = a2 - 0.5 * numerator / denominator

        # update model
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p -= alpha_star * g

        self.total_loss += loss.item()
        self.optimize_step += 1

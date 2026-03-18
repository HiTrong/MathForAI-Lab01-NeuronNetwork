from optimizers.gradient_descent import MySGD
from optimizers.parabolic import Parabolic
from models.neuron_network import SimpleMLP
from tqdm.auto import tqdm
from utils.data import get_mnist_data
from utils.metrics import evaluate_accuracy
import matplotlib.pyplot as plt
import torch.nn as nn
import logging
import time

# Logging
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# Dataset
train_loader, val_loader = get_mnist_data()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Model & Optimizer
sgd_model = SimpleMLP()
sgd_optimizer = MySGD(
    loss_function=criterion,
    parameters=sgd_model.parameters(),
    lr=0.01
)

parabolic_model = SimpleMLP()
parabolic_optimizer = Parabolic(
    loss_function=criterion,
    lr=0.01
)

EPOCHS = 100
# Training for sgd model
sgd_train_loss = []
sgd_val_loss = []
sgd_val_acc = []

print("[START] TRAINING FOR SGD MODEL\n")

for epoch in range(EPOCHS):

    sgd_model.train()
    sgd_optimizer.reset()

    time_start = time.time()

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - SGD Training"):

        outputs = sgd_model(images)

        sgd_optimizer.optimize(
            outputs,
            labels
        )

    train_loss = sgd_optimizer.total_loss / sgd_optimizer.optimize_step

    val_loss, val_acc = evaluate_accuracy(
        sgd_model,
        val_loader,
        criterion
    )

    time_end = time.time()

    message = (
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}% | "
        f"Time: {(time_end-time_start):.2f}s"
    )

    sgd_train_loss.append(train_loss)
    sgd_val_loss.append(val_loss)
    sgd_val_acc.append(val_acc)

    print(message)
    logging.info("[SGD] " + message)

print("\n[END] TRAINING FOR SGD MODEL\n\n")


# Training for parabolic model
para_train_loss = []
para_val_loss = []
para_val_acc = []

print("[START] TRAINING FOR PARABOLIC MODEL\n")

for epoch in range(EPOCHS):

    parabolic_model.train()
    parabolic_optimizer.reset()

    time_start = time.time()

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Parabolic Training"):
        
        parabolic_optimizer.optimize(
            parabolic_model,
            images,
            labels
        )

    train_loss = parabolic_optimizer.total_loss / parabolic_optimizer.optimize_step

    val_loss, val_acc = evaluate_accuracy(
        parabolic_model,
        val_loader,
        criterion
    )

    time_end = time.time()

    message = (
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}% | "
        f"Time: {(time_end-time_start):.2f}s"
    )

    para_train_loss.append(train_loss)
    para_val_loss.append(val_loss)
    para_val_acc.append(val_acc)

    print(message)
    logging.info("[PARABOLIC] " + message)

print("\n[END] TRAINING FOR PARABOLIC MODEL\n\n")

# Visualize Loss
plt.figure()

plt.plot(sgd_train_loss, label="SGD Train Loss")
plt.plot(sgd_val_loss, label="SGD Val Loss")

plt.plot(para_train_loss, label="Parabolic Train Loss")
plt.plot(para_val_loss, label="Parabolic Val Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend()

plt.savefig("loss_comparison.png")
plt.close()

# Visualize Accuracy
plt.figure()

plt.plot(sgd_val_acc, label="SGD")
plt.plot(para_val_acc, label="Parabolic")

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy Comparison")
plt.legend()

plt.savefig("accuracy_comparison.png")
plt.close()
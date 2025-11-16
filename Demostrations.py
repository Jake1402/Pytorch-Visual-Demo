import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from model import Network
from functions import *

def return_model(inputs = 1, nodes=128, outputs=1, activation = torch.nn.Tanh):
    ann = Network(
        inputs=inputs,
        nodes=nodes,
        output=outputs,
        activation=activation
    )
    print(f"Model parameters : {sum(param.numel() for param in ann.parameters() if param.requires_grad)}")
    return ann

def configure_training(lr, ann, batch_size, X, Y):
    optim = torch.optim.Adam(ann.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    tensor_ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))
    tensor_dl = DataLoader(tensor_ds, batch_size=batch_size, shuffle=True)
    return (optim, loss_fn, tensor_dl)

def train(epochs, dl, loss_fn, optim, ann):
    for epoch in range(epochs):
        t_loss, ds_size = 0, 0
        for batch, (X, y) in enumerate(dl):
            ds_size += X.shape[0]
            optim.zero_grad()
            preds = ann(X.unsqueeze(1))
            loss = loss_fn(preds, y.unsqueeze(1))
            t_loss += loss
            loss.backward()
            optim.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1} over, average loss {t_loss/ds_size :.4f}")

def generate_data(func, samples, _range):
    x_data = np.linspace(start=-_range, stop=_range, num=samples)
    x_data_train   = np.concatenate([x_data, x_data], axis=0)
    clean_function = func(x_data_train)
    function_train = clean_function + (np.random.random(clean_function.shape)/5)-0.1
    return x_data, func(x_data), x_data_train, function_train

def plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func, axs):
    ann = return_model(inputs=inputs, nodes=nodes, outputs=outputs, activation=activation)
    x_data, func_real, x_data_train, function_train = generate_data(func=func, samples=samples, _range=_range)

    axs.set_title(f"Model with {nodes} hidden units leaning {func.name}", size=10)
    axs.scatter(x_data_train, function_train, 0.3, "g")
    axs.plot(x_data, func_real)

    optim, loss_fn, tensor_dl = configure_training(lr=lr, ann=ann, batch_size=batch_size, X=x_data_train, Y=function_train)
    train(epochs=epochs, dl=tensor_dl, loss_fn=loss_fn, optim=optim, ann=ann)
    predictions = ann(torch.as_tensor(x_data, dtype=torch.float32).unsqueeze(1)).squeeze(1)

    axs.plot(x_data, predictions.detach().numpy())
    axs.legend(["Noisy", func.name, "Prediction"])
    del ann

def plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func):
    ann = return_model(inputs=inputs, nodes=nodes, outputs=outputs, activation=activation)
    x_data, func_real, x_data_train, function_train = generate_data(func=func, samples=samples, _range=_range)
    
    plt.title(f"Model with {nodes} hidden units leaning {func.name}", size=10)
    plt.scatter(x_data_train, function_train, 0.3, "g")
    plt.plot(x_data, func_real)

    optim, loss_fn, tensor_dl = configure_training(lr=lr, ann=ann, batch_size=batch_size, X=x_data_train, Y=function_train)
    train(epochs=epochs, dl=tensor_dl, loss_fn=loss_fn, optim=optim, ann=ann)
    predictions = ann(torch.as_tensor(x_data, dtype=torch.float32).unsqueeze(1)).squeeze(1)

    plt.plot(x_data, predictions.detach().numpy())
    plt.legend(["Noisy", func.name, "Prediction"])
    plt.savefig(f"./images/Model with {nodes} hidden units leaning {func.name}")
    plt.close()
    del ann


if __name__ == "__main__":

    inputs = 1
    nodes=128
    outputs=1
    activation = torch.nn.ReLU

    lr = 0.001
    epochs = 100
    batch_size = 8

    samples=250
    _range=2*np.pi

    func_quad = quad_func()
    func_sinx = sinx_func()
    func_cosx = cosx_func()

    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_quad)
    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_sinx)
    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_cosx)
    nodes=8
    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_quad)
    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_sinx)
    plot_single(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_cosx)

    nodes=128
    fig, axs = plt.subplots(2, 3)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
    fig.suptitle("Showing how simple ANNs can model different functions")
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_quad, axs[0,0])
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_sinx, axs[0,1])
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_cosx, axs[0,2])

    nodes=8
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_quad, axs[1,0])
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_sinx, axs[1,1])
    plot(inputs, nodes, outputs, activation, lr, epochs, batch_size, samples, _range, func_cosx, axs[1,2])
    #plt.savefig("./images/Single Figure Comparison")
    plt.show()
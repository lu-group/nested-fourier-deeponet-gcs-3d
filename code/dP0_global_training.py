import os
os.environ['DDE_BACKEND'] = 'pytorch'
from deepxde.nn import activations
from deepxde.nn.pytorch.fnn import FNN
import deepxde as dde
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MIONetCartesianProd(dde.nn.pytorch.NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
            self,
            layer_sizes_branch1,
            layer_sizes_branch2,
            layer_sizes_trunk,
            activation,
            kernel_initializer,
            regularization=None,
            trunk_last_activation=False,
            merge_operation="mul",
            layer_sizes_merger=None,
            output_merge_operation="mul",
            layer_sizes_output_merger=None
    ):
        super().__init__()

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch1[1]):
            # User-defined network
            self.branch1 = layer_sizes_branch1[1]
        else:
            # Fully connected network
            self.branch1 = FNN(
                layer_sizes_branch1, self.activation_branch1, kernel_initializer
            )
        if callable(layer_sizes_branch2[1]):
            # User-defined network
            self.branch2 = layer_sizes_branch2[1]
        else:
            # Fully connected network
            self.branch2 = FNN(
                layer_sizes_branch2, self.activation_branch2, kernel_initializer
            )
        if layer_sizes_merger is not None:
            self.activation_merger = activations.get(activation["merger"])
            if callable(layer_sizes_merger[1]):
                # User-defined network
                self.merger = layer_sizes_merger[1]
            else:
                # Fully connected network
                self.merger = FNN(
                    layer_sizes_merger, self.activation_merger, kernel_initializer
                )
        else:
            self.merger = None
        if layer_sizes_output_merger is not None:
            self.activation_output_merger = activations.get(activation["output merger"])
            if callable(layer_sizes_output_merger[1]):
                # User-defined network
                self.output_merger = layer_sizes_output_merger[1]
            else:
                # Fully connected network
                self.output_merger = FNN(
                    layer_sizes_output_merger, self.activation_output_merger, kernel_initializer
                )
        else:
            self.output_merger = None
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.trunk_last_activation = trunk_last_activation
        self.merge_operation = merge_operation
        self.output_merge_operation = output_merge_operation

    def forward(self, inputs):
        x_func1 = inputs[0]
        # x_func2 = inputs[1]
        x_loc = inputs[2]

        # Branch net to encode the input function
        y_func = self.branch1(x_func1)
        # y_func1 = self.branch1(x_func1)
        # y_func2 = self.branch2(x_func2)
        # y_func2 = y_func2.reshape(y_func2.shape[0], y_func2.shape[1], 1, 1, 1)
        # if self.merge_operation == "sum":
        #     x_merger = y_func1 + y_func2
        # elif self.merge_operation == "mul":
        #     x_merger = torch.mul(y_func1, y_func2)
        # else:
        #     raise NotImplementedError(
        #         f"{self.merge_operation} operation to be implimented"
        #     )
        # # Optional merger net
        # if self.merger is not None:
        #     y_func = self.merger(x_merger)
        # else:
        #     y_func = x_merger

        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        # trunk net
        y_loc = self.trunk(x_loc)

        if self.trunk_last_activation:
            y_loc = self.activation_trunk(y_loc)

        # output merger net
        if self.output_merger is None:
            y = torch.einsum("ip,jp->ij", y_func, y_loc)
        else:
            y_func = y_func[:, None, :, :, :, :]
            y_loc = y_loc[None, :, :, None, None, None]
            if self.output_merge_operation == "mul":
                y = torch.mul(y_func, y_loc)
            elif self.output_merge_operation == "sum":
                y = y_func + y_loc
            batch_size = y.shape[0]
            timestep_size = y.shape[1]
            channal_num = y.shape[2]
            x_size = y.shape[3]
            y_size = y.shape[4]
            z_size = y.shape[5]
            y = y.reshape(batch_size * timestep_size, channal_num, x_size, y_size, z_size)
            y = self.output_merger(y)
            x_size = y.shape[1]
            y_size = y.shape[2]
            z_size = y.shape[3]
            y = y.reshape(batch_size, timestep_size, x_size, y_size, z_size, 1)
        # Add bias
        # y = y + self.b
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y

class QuadrupleCartesianProd(dde.data.Data):
    """Cartesian Product input data format for MIONet architecture.

    This dataset can be used with the network ``MIONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays. The first element has the shape (`N1`,
            `dim1`), the second element has the shape (`N1`, `dim2`), and the third
            element has the shape (`N2`, `dim3`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test, time_batch_size, time_steps):
        # if (
        #     len(X_train[0]) * len(X_train[2]) != y_train.size
        #     or len(X_train[1]) * len(X_train[2]) != y_train.size
        #     or len(X_train[0]) != len(X_train[1])
        # ):
        #     raise ValueError(
        #         "The training dataset does not have the format of Cartesian product."
        #     )
        # if (
        #     len(X_test[0]) * len(X_test[2]) != y_test.size
        #     or len(X_test[1]) * len(X_test[2]) != y_test.size
        #     or len(X_test[0]) != len(X_test[1])
        # ):
        #     raise ValueError(
        #         "The testing dataset does not have the format of Cartesian product."
        #     )
        self.indices_timestep = None                    ###
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.time_batch_size = time_batch_size          ###
        self.time_steps = time_steps                    ###

        self.train_sampler = dde.data.sampler.BatchSampler(len(X_train[0]), shuffle=True)
        self.train_timestep_sampler = dde.data.sampler.BatchSampler(self.time_steps, shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        self.indices_timestep = self.train_timestep_sampler.get_next(self.time_batch_size)
        size = self.train_y.shape[0]
        return (
                   self.train_x[0][indices],
                   self.train_x[1][indices],
                   self.train_x[2][self.indices_timestep],
               ), self.train_y[indices, :][:, self.indices_timestep, :]

    def test(self):
        return (self.test_x[0], self.test_x[1], self.test_x[2][self.indices_timestep],), self.test_y[:, self.indices_timestep, :]

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class decoder(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(decoder, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.width2 = width * 4
        self.padding = 8

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, self.width2)
        self.fc2 = nn.Linear(self.width2, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2

        x = x[:, :, self.padding * 2:-self.padding * 2,
            self.padding * 2:-self.padding * 2, self.padding:-self.padding]

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

class branch1(nn.Module):
    def __init__(self, width):
        super(branch1, self).__init__()
        self.width = width
        self.padding = 8
        self.fc0 = nn.Linear(7, self.width)

    def forward(self, x):
        # batchsize = x.shape[0]
        # size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [self.padding, self.padding, self.padding * 2, self.padding * 2, self.padding * 2,
                      self.padding * 2])
        return x

class branch2(nn.Module):
    def __init__(self, width):
        super(branch2, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(1, self.width)

    def forward(self, x):
        x = self.fc0(x)

        return x

gelu = torch.nn.GELU()

width = 32
Net = MIONetCartesianProd(
    layer_sizes_branch1=[10 * 100 * 100 * 5, branch1(width)], layer_sizes_branch2=[3 * 28, branch2(width)], #branch2 is not used
    layer_sizes_trunk=[1, 100, 100, 100, width],
    activation={"branch1": gelu, "branch2": gelu, "trunk": gelu, "merger": gelu, "output merger": gelu},
    kernel_initializer="Glorot normal",
    # regularization=("l2", 4e-6),
    trunk_last_activation=False,
    merge_operation="mul",
    layer_sizes_merger=None,
    output_merge_operation="mul",
    layer_sizes_output_merger=[5, decoder(20, 20, 2, width)])

ntrain = 2407
nval = 1

t = np.load('../datasets/trunk_input.npy').astype(np.float32)

x_train = np.load('./dP_GLOBAL_train_input.npz')['input'][:ntrain].astype(np.float32)
x_train[..., -1] = (x_train[..., -1] - 1.1501) / (0.9758)
x_train_MIO = np.load('./dP_GLOBAL_train_input.npz')['input'][:ntrain, 0, 0, 0, 5:6].astype(np.float32) #Not used
x_train = (x_train, x_train_MIO, t)
mean = torch.from_numpy(np.load('./dP0_outputs_mean_std.npz')['mean']).cuda()
std = torch.from_numpy(np.load('./dP0_outputs_mean_std.npz')['std']).cuda()
y_train = np.moveaxis(np.load('./dP_GLOBAL_train_output.npz')['output'][:ntrain], 4, 1).astype(np.float32)

x_test = np.load('./dP_GLOBAL_val_input.npz')['input'][-nval:].astype(np.float32)
x_test[..., -1] = (x_test[..., -1] - 1.1501) / (0.9758) #Normalization
x_test_MIO = np.load('./dP_GLOBAL_val_input.npz')['input'][-nval:, 0, 0, 0, 5:6].astype(np.float32)
x_test = (x_test, x_test_MIO, t)
y_test = np.moveaxis(np.load('./dP_GLOBAL_val_output.npz')['output'][-nval:], 4, 1).astype(np.float32)

time_batch = 6

data = QuadrupleCartesianProd(x_train, y_train, x_test, y_test, time_batch, 24)

def rel(y_true, y_pred):
    indices_timestep = data.indices_timestep
    y_pred = (y_pred * std[:, indices_timestep]) + mean[:, indices_timestep]
    num_examples = y_pred.size()[0]

    diff_norms = torch.norm(y_pred.reshape(num_examples, -1) - y_true.reshape(num_examples,-1), 2, 1)
    y_norms = torch.norm(y_true.reshape(num_examples, -1), 2, 1)

    return torch.mean(diff_norms/y_norms)

model = dde.Model(data, Net)
num_epochs = 3

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

path = './dP0_saved_models'
mkdir(path)

start_time = time.time()

model.compile("adam", loss=rel, lr=1e-3, decay=("step", ntrain*2*round(24/time_batch), 0.9))
checker = dde.callbacks.ModelCheckpoint(f"{path}/model.ckpt", save_better_only=False, period=ntrain*round(24/time_batch))
losshistory, train_state = model.train(epochs=ntrain*round(24/time_batch)*num_epochs, batch_size=1, display_every=ntrain*round(24 / time_batch), callbacks=[checker])

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.4f} seconds")
print(model.net.num_trainable_parameters())
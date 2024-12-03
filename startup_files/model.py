import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):

    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))


        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))

    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer_neurons = 10

        # First Convolutional Layer
        self.layer1_filters = 10
        self.layer1_kernel_size = (4, 4)
        self.layer1_stride = 1
        self.layer1_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) // self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) // self.layer1_stride + 1

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer1_padding)

        

        # Compute input size for fully connected layer
        self.fc_inputs = int(self.layer1_filters * self.layer1_dim_h * self.layer1_dim_w)

        # Fully connected layers
        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)
        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        
        # Print the shape before flattening
        print(x.shape)

        # Flatten the convolutional output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class CNN_small(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer_neurons = 200

        # First Convolutional Layer
        self.layer1_filters = 32
        self.layer1_kernel_size = (4, 4)
        self.layer1_stride = 1
        self.layer1_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) // self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) // self.layer1_stride + 1

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer1_padding)

        # Second Convolutional Layer
        self.layer2_filters = 64
        self.layer2_kernel_size = (3, 3)
        self.layer2_stride = 1
        self.layer2_padding = 0

        self.layer2_dim_h = (self.layer1_dim_h - self.layer2_kernel_size[0]) // self.layer2_stride + 1
        self.layer2_dim_w = (self.layer1_dim_w - self.layer2_kernel_size[1]) // self.layer2_stride + 1

        self.conv2 = nn.Conv2d(self.layer1_filters, self.layer2_filters, self.layer2_kernel_size, stride=self.layer2_stride, padding=self.layer2_padding)

        # Third Convolutional Layer
        self.layer3_filters = 128
        self.layer3_kernel_size = (3, 3)
        self.layer3_stride = 1
        self.layer3_padding = 1  # Adding padding to maintain spatial dimensions

        self.layer3_dim_h = self.layer2_dim_h
        self.layer3_dim_w = self.layer2_dim_w

        self.conv3 = nn.Conv2d(self.layer2_filters, self.layer3_filters, self.layer3_kernel_size, stride=self.layer3_stride, padding=self.layer3_padding)

        # Fourth Convolutional Layer
        self.layer4_filters = 128
        self.layer4_kernel_size = (3, 3)
        self.layer4_stride = 2
        self.layer4_padding = 1  # Adding padding

        self.layer4_dim_h = (self.layer3_dim_h - self.layer4_kernel_size[0] + 2 * self.layer4_padding) // self.layer4_stride + 1
        self.layer4_dim_w = (self.layer3_dim_w - self.layer4_kernel_size[1] + 2 * self.layer4_padding) // self.layer4_stride + 1

        self.conv4 = nn.Conv2d(self.layer3_filters, self.layer4_filters, self.layer4_kernel_size, stride=self.layer4_stride, padding=self.layer4_padding)

        # Compute input size for fully connected layer
        self.fc_inputs = int(self.layer4_filters * self.layer4_dim_h * self.layer4_dim_w)

        # Fully connected layers
        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer_neurons)
        self.lin2 = nn.Linear(self.fc_layer_neurons, self.out_dim)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten the convolutional output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
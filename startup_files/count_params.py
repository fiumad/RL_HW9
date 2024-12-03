import torch
from model import CNN # Replace with your actual model class and file

IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

resize_factor = 1
new_h = int(IMAGE_HEIGHT / resize_factor)
new_w = int(IMAGE_WIDTH / resize_factor)

input_dim = (3, new_h, new_w)
output_dim = 11

# Example: Replace 'your_model.pth' with your model's path
model_path = './models/cnn_buildings.pth'

# Define the model architecture
model = CNN(in_dim=input_dim, out_dim=output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"The model has {total_params} parameters.")



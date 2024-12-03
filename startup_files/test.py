import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNN
from dataset import BuildingDataset

IMAGE_HEIGHT = 189
IMAGE_WIDTH = 252

def evaluate_model(model, loader, device):
    #Evaluate the model on a dataset and calculate accuracy.
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Get predictions
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def main(dataset_path, labels_file):
    #Main function to load the model, prepare the dataset, and evaluate accuracy.
    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        sys.exit(1)

    # Ensure labels file exists
    labels_path = os.path.join(dataset_path, labels_file)
    if not os.path.exists(labels_path):
        print(f"Error: Labels file {labels_path} does not exist.")
        sys.exit(1)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image preprocessing transforms
    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the new dataset
    new_dataset = BuildingDataset(labels_path, dataset_path, transform=test_transforms)
    data_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)

    # Load the model
    input_dim = (3, IMAGE_HEIGHT, IMAGE_WIDTH)
    out_dim = 11  # Ensure this matches the number of classes
    model = CNN(in_dim=input_dim, out_dim=out_dim).to(device)
    model_path = "models/cnn_buildings.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    # Evaluate the model
    evaluate_model(model, data_loader, device)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <path_to_new_dataset> <labels_file>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    labels_file = sys.argv[2]
    main(dataset_path, labels_file)


#Imports here
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import os

# Model Loading
def load_model(arch, hidden_units):
    """Load the model based on architecture name and adjust the classifier."""
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(in_features, hidden_units)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(0.5)),
                ("fc2", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1))
            ])
        )
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(in_features, hidden_units)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(0.5)),
                ("fc2", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1))
            ])
        )
    else:
        raise ValueError("Supported architectures: 'resnet18', 'vgg13'")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    return model

# Training Model
def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    """Train the neural network model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Data transforms and loaders for training and validation sets
    train_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    valid_data = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=valid_transforms)

    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)

    # Load the model
    model = load_model(arch, hidden_units)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Training the model
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()  # Set model to evaluation mode
                valid_loss = 0
                accuracy = 0

                # Validate the model
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}"
                )

                running_loss = 0
                model.train()  # Set model back to training mode

    # Save the checkpoint
    checkpoint = {
        "arch": arch,
        "epoch": epochs,
        "state_dict": model.state_dict(),
        "class_to_idx": train_data.class_to_idx,
        "hidden_units": hidden_units,
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))
    print(f"Checkpoint saved to {save_dir}/checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    
    # Positional arguments
    parser.add_argument("data_dir", help="Directory containing the data")

    # Optional arguments
    parser.add_argument("--save_dir", help="Directory to save checkpoints", default=".")
    parser.add_argument("--arch", help="Architecture of the model", default="resnet18")
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=0.003)
    parser.add_argument("--hidden_units", help="Number of hidden units", type=int, default=512)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=5)
    parser.add_argument("--gpu", help="Use GPU for training", action="store_true")
    
    args = parser.parse_args()

    train(
        args.data_dir,
        args.save_dir,
        args.arch,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
        args.gpu,
    )

#Inports here
import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

# Checkpoint Loading
def load_checkpoint(checkpoint_path):
    """
    Load the model from a checkpoint file and reconstruct the architecture.
    """
    checkpoint = torch.load(checkpoint_path)

    # Rebuild the model architecture based on the checkpoint
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"])),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(checkpoint["hidden_layers"], checkpoint["output_size"])),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    # Load the saved model state
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image):
    """
    Process the input image to the format expected by the model.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return preprocess(image).unsqueeze(0)  # Add batch dimension

#Image prediction of top 5 classes
def predict(image_path, model, topk=5, gpu=False):
    """
    Predict the class of an image using a trained model.
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and process the image
    image = Image.open(image_path)
    image = process_image(image).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(topk)

        # Convert to numpy arrays for easier handling
        probs = probs.cpu().numpy().flatten()
        indices = indices.cpu().numpy().flatten()

        # Convert indices to class names using class_to_idx mapping
        idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
        indices = [idx_to_classes[idx] for idx in indices]

    return probs, indices


def load_category_names(json_file):
    """
    Load category names from a JSON file.
    """
    with open(json_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser(description="Predict the class of a flower image")
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("checkpoint", help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", help="JSON file mapping category indices to flower names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load the model from checkpoint
    model = load_checkpoint(args.checkpoint)

    # Predict top K classes
    probs, indices = predict(args.image_path, model, args.top_k, args.gpu)

    # Load category names if provided, otherwise display class indices
    if args.category_names:
        category_names = load_category_names(args.category_names)
        output_names = [category_names.get(str(idx), "Unknown Flower") for idx in indices]
    else:
        output_names = indices

    # Print the results
    for i in range(args.top_k):
        print(f"Class: {output_names[i]}, Probability: {probs[i]:.4f}")

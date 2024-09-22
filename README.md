# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


# How to Run train.py

Run the script with:
python train.py data_directory
This will train a model on the dataset located in the data_directory with default settings.

Set the directory to save checkpoints:

Use the --save_dir option to specify where to save the trained model checkpoint:

python train.py data_directory --save_dir save_directory

Choose the architecture:

Use the --arch option to specify the model architecture (either resnet18 or vgg13):
python train.py data_directory --arch "vgg13"

Set hyperparameters:

You can set the learning rate, number of hidden units, and number of epochs with:
python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20

Use GPU for training:
Add --gpu to train using GPU if available:
python train.py data_directory --gpu

# For running predict.py
One can run the script with python predict.py /path/to/image checkpoint to predict the flower name and class probability for an image.

The Image classifier is built on google collab. It can be directly run on it after saving its copy to drive.
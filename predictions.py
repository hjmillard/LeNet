# Program for loading the trained model and making predictions of a number in the 32x32 pixel image

import torch
from torchvision import transforms
from torch import nn
from PIL import Image

# Model Structure
CNN = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=(1,1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=(1,1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(16*7*7, 10)
)

# Loading the pretrained model into RAM
CNN.load_state_dict(torch.load('model_state.pt'))
CNN.eval()

# Making predictions
input_file = None
format_image = transforms.Compose([transforms.ToTensor()])
while input_file is None or input_file != 'exit':
    try:
        input_file = input('Please enter a filepath:\n')
        if input_file == 'exit':
            break
        img = Image.open(input_file)
        with torch.no_grad():
            output = CNN(format_image(img).unsqueeze(0))
        _, prediction = torch.max(output, 1)
        print('Classifier:', prediction.item())
    except:
        print('An error occurred. Ensure you have entered the correct path to the image file. Please try again. ')

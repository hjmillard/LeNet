# LeNet
Modified LeNet using DropOut, ReLU and Max Pooling Layers.
The model takes in a 32x32 pixel image and makes a prediction as to which number (0-9) it 'thinks' is in the image.

To make predictions, run the following command:
    python predictions.py
This will load the model into RAM and prompt the user to enter the path to a 32x32 pixel image which the program will make predictions on.

To train the model, run the following command:
    python classifier.py
This will begin training the model, after which, the user will be prompted to enter an image to predict.
The image must be 32x32 pixels and should contain a number (0-9).

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image


# Loading the data from disk
transform = transforms.Compose([transforms.ToTensor()])
directory = '.'
train_data = datasets.MNIST(directory, train=True, download=False, transform=transform)
test_data = datasets.MNIST(directory, train=False, download=False, transform=transform)
train_set, validation_set = data.random_split(train_data, (48000, 12000))

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

cost_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters())

def calc_accuracy(logits, expected):
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

if __name__ == '__main__':
    print('Training model...')
    train_loss = []
    val_accuracies = []
    best_model = None
    max_accuracy = None
    best_epoch = None
    max_epoch = 100
    no_improvement = 5
    batch_size = 512

    log_file = open("log.txt", "w")
    for epoch in range(max_epoch):
        # set model to train mode
        CNN.train()
        loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
        epoch_loss = []
        # train on each image in the loaded batch
        for x_i, y_i in loader:
            optimizer.zero_grad()
            prediction = CNN(x_i)
            loss = cost_function(prediction, y_i)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach())
            log_file.write(f'Epoch [{epoch}/{max_epoch}], Loss: {loss.item():.4f}\n')
        train_loss.append(torch.tensor(epoch_loss).mean())
        # set model to evaluation mode
        CNN.eval()
        # make prediction
        loader = data.DataLoader(validation_set, batch_size=len(validation_set), shuffle=False)
        X, y = next(iter(loader))
        prediction = CNN(X)
        # record prediction accuracy
        current_accuracy = calc_accuracy(prediction, y).detach()
        val_accuracies.append(current_accuracy)
        if max_accuracy is None or current_accuracy > max_accuracy:
            print("New best epoch ", epoch, "acc", current_accuracy)
            max_accuracy = current_accuracy
            best_model = CNN.state_dict()
            best_epoch = epoch
            log_file.write(f'Validation Accuracy: {max_accuracy.item():.4f}')
        if best_epoch + no_improvement <= epoch:
            print("Terminating training. Model hasn't improved for", no_improvement, "epochs")
            log_file.write(f'Final Validation Accuracy: {max_accuracy.item():.4f}')
            break
    log_file.close()
    torch.save(CNN.state_dict(), 'model_state.pt')
    CNN.load_state_dict(best_model)
    CNN.eval()
    
    # taking user input and making predictions
    print('Done training model...')
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
    
    plt.title('Validation accuracy. Dot denotes best accuracy.')
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.plot(best_epoch, max_accuracy, 'bo', label='Best accuracy')
    plt.show()
    plt.title('Training loss')
    plt.plot(train_loss)
    plt.show()
    k = max(3*no_improvement, 0)
    plt.title('Last {} epochs'.format(k))
    plt.plot(val_accuracies[-k:])
    plt.plot(best_epoch-(len(val_accuracies)-k), max_accuracy, 'bo')
    plt.show()

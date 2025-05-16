

import torch
import torch.nn as nn
import torchvision.models as models
from vocr_dataset import VCoR


class CustomWideResNet101(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomWideResNet101, self).__init__()
        # Load pre-trained WideResNet50-2 model
        self.wide_resnet = models.wide_resnet101_2(pretrained=True)
        
        # Modify the final fully connected layer to output 6 classes
        in_features = self.wide_resnet.fc.in_features
        self.wide_resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.wide_resnet(x)

# Instantiate the custom model
model = CustomWideResNet101()

model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train_dataset = VCoR('/data/sunzc/VCoR/train_4_25')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2560, shuffle=True)


test_dataset = VCoR('/data/sunzc/VCoR/test_v2')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=320, shuffle=False)



num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

for epoch in range(num_epochs):
    iteration = 0
    for images, labels, image_name in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iteration [{iteration+1}], Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            
        iteration += 1
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



    if (epoch+1) % 20 == 0:
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels, image_name in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(labels, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy*100:.2f}%")

        torch.save(model.state_dict(), f'/home/sunzc/chatgpt/color_classification/wide_resnet_color_model_{epoch}_{accuracy*100:.2f}.pth')
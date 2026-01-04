import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from multiprocessing import freeze_support
from tqdm import tqdm

if __name__ == '__main__':
    freeze_support()

    # Load the pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        if "fc" in name:  # Unfreeze the final classification layer
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(nn.Flatten(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1),
    nn.Sigmoid())


    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the model for testing
    model.load_state_dict(torch.load('bubble_load_model_cut_unfreezed.pth'))
    model.eval()

    # Define the test data directory and transformations
    test_data_dir = './Dataset_cut/Test'
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

        # Create the test dataset and dataloader
    test_dataset = datasets.ImageFolder(test_data_dir, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

        # Function to add noise to an image, excluding pixels that are 0 or 255
    def add_noise(img, mean=0, std=0.1):
        noise = torch.randn(img.size()) * std + mean
        mask = (img != img.min()) & (img != img.max())
        noisy_img = img + noise * mask.float()
        return torch.clamp(noisy_img, 0, 1)

    ################## Noise #########################
    ### This part is to add noise to the images, making them blur
    ### Comment this part to train model with no noise
    # Apply noise to each image in the dataset
    # noisy_datasets = {'test': []}
    # for phase in ['test']:
    #     for inputs, labels in test_dataloader:
    #         noisy_inputs = []
    #         for img in inputs:
    #             noisy_img = add_noise(img)
    #             noisy_inputs.append(noisy_img)
    #         noisy_datasets[phase].append((torch.stack(noisy_inputs).squeeze(0), labels))

    # dataloaders = {x: torch.utils.data.DataLoader(noisy_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['test']}
    # ################## End #############################

    # Evaluate the model on the test data
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1, 1).float()

            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_dataset)
    print(f'Test Acc: {test_acc:.4f}')
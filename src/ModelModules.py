# -*- coding: utf-8 -*-
"""
@Author: Thanasis
"""

import torch
from tqdm import tqdm
from Dataset import train_dataset, test_dataset

def train_model(model, optimizer, criterion, scheduler, device, train_loader, num_epochs, PATH=None, valid_loader=None):
    for epoch in range(num_epochs):
        correct = 0
        for inputs, labels in (tqdm(train_loader)):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            classifications = torch.argmax(outputs, dim=1)
            correct += (classifications == labels).sum()
            del inputs, labels, outputs, classifications
            # Backward pass
            loss.backward()
            optimizer.step()

        # Print the loss for every epoch
        if scheduler is not None:
            scheduler.step()
        
        # Calculate and print the average accuracy for every epoch
        accuracy = 100 * correct / len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
        
        # Print the validation accuracy for every 10 epochs
        if (valid_loader is not None and (epoch+1)%10==0):
            print("Validation:")
            test_model(model, device, valid_loader)
    
    # Save the model depending on whether it is a DataParallel model or not
    if PATH is not None:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), PATH)
        else:
            torch.save(model.state_dict(), PATH)
        
def test_model(model, device, test_loader, PATH=None):
    # Load model depending on current device
    if PATH is not None:
        if device == 'cpu':
            model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(PATH))
            
    # Set model to evaluation mode
    model.eval()
    correct = 0
    
    # Calculate the accuracy of the model
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            classifications = torch.argmax(outputs, dim=1)
            correct += (classifications == labels).sum()
            del inputs, labels, outputs, classifications
    
    accuracy = 100 * correct / len(test_dataset)
    print(f'Accuracy: {accuracy:.4f}')
        

from tqdm import tqdm
import torch


def train_epoch(network, loss_fn, dataloader, optimizer, device = 'cpu'):
       # Set the network to train mode
    network.to(device)
    network.train()
    
    # Initialize variables to keep track of loss and number of batches
    epoch_loss = 0.0
    num_batches = 0
    epoch_correct = 0.0
    num_samples = 0

    
    # Iterate over the data loader
    for batch_inputs, batch_targets in dataloader:
        # Move data to the appropriate device (e.g., GPU)
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        batch_outputs = network(batch_inputs)
        
        # Compute the loss
        loss = loss_fn(batch_outputs, batch_targets)
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        # Accumulate the loss
        epoch_loss += loss.item()
        num_batches += 1

       # Calculate the number of correct predictions in the batch
        _, predicted = torch.max(batch_outputs, 1)
        epoch_correct += (predicted == batch_targets).sum().item()
        num_samples += batch_targets.size(0)

    
    # Calculate the average loss for the epoch
    average_loss = epoch_loss / num_batches

    # Calculate the accuracy for the epoch
    accuracy = epoch_correct / num_samples

    
    return average_loss, accuracy


def validate_epoch(network, loss_fn, dataloader, device = 'cpu'):
    # Set the network to evaluation mode
    network.to(device)
    network.eval()
    
    
    # Initialize variables to keep track of loss and number of batches
    epoch_loss = 0.0
    num_batches = 0
    epoch_correct = 0.0
    num_samples = 0
    
    # Turn off gradients
    with torch.no_grad():
        # Iterate over the data loader
        for batch_inputs, batch_targets in dataloader:
            # Move data to the appropriate device (e.g., GPU)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            batch_outputs = network(batch_inputs)
            
            # Compute the loss
            loss = loss_fn(batch_outputs, batch_targets)
            
            # Accumulate the loss
            epoch_loss += loss.item()
            num_batches += 1

            # Calculate the number of correct predictions in the batch
            _, predicted = torch.max(batch_outputs, 1)
            epoch_correct += (predicted == batch_targets).sum().item()
            num_samples += batch_targets.size(0)
    
    # Calculate the average loss for the epoch
    average_loss = epoch_loss / num_batches

    # Calculate the accuracy for the epoch
    accuracy = epoch_correct / num_samples
    
    return average_loss, accuracy


def train(network, loss_fn, train_dataloader, val_dataloader, optimizer, num_epochs, device='cpu'):
    train_losses = []
    val_losses = []
    val_acc = []
    train_acc = []

    best_val_acc = 0.0
    best_epoch = -1

    # Initialize tqdm for progress tracking
    progress_bar = tqdm(range(num_epochs), desc='Training Progress')
 
    for epoch in progress_bar:
        # Training phase
        train_loss, train_accuracy = train_epoch(network, loss_fn, train_dataloader, optimizer, device=device)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        # Validation phase
        val_loss, val_accuracy = validate_epoch(network, loss_fn, val_dataloader,device=device)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch
            #print(f"Best val acc {best_val_acc:.3f} at epoch {epoch}")
        # Print the loss for each epoch
        #print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},Val Loss: {val_loss:.4f} Val Acc: {accuracy:.4f}')
        
        # Update tqdm progress bar
        progress_bar.set_postfix({'Train Loss':train_loss, 
                                  'Val Loss':val_loss, 
                                  'Train Acc': train_accuracy, 
                                  'Val Acc': val_accuracy, 
                                  'Best Val Acc':best_val_acc})
                                  
    return train_losses, val_losses, train_acc, val_acc 
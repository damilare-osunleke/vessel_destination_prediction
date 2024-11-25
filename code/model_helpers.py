import torch
import torch.nn as nn


def train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        batch_size = x_batch.size(0)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 1000 == 0: 
            current_loss = total_loss / total_samples
            print(f'Training - Epoch: {epoch+1}, Batch: {batch_index+1}, Current Average Loss: {current_loss:.4f}')

    average_loss = total_loss / total_samples
    return average_loss



def validate_one_epoch(model, val_loader, loss_function, device, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(val_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        batch_size = x_batch.size(0)

        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 1000 == 0:
            current_loss = total_loss / total_samples
            print(f'Validation - Epoch: {epoch+1}, Batch: {batch_index+1}, Current Average Loss: {current_loss:.4f}')

    average_loss = total_loss / total_samples
    return average_loss



class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), '../output/models/best_model_val_loss.pt')  # Save the best model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  

from src.evaluation import evaluate


class YawDDclassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # define layer
        self.fc1 = nn.Linear(376320, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x.squeeze(-1)
    

def trainer(trainloader,
            valloader,
            model,
            epochs,
            lr,
            device):

    # optimizer 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in range(epochs):

        # init running loss
        running_loss = 0

        # go through all data
        model.train()
        for frames, labels in tqdm(trainloader, desc=f'Epoch {epoch}'):
            frames, labels = frames.to(device), labels.to(device) # shift data to device

            # forward + backward pass
            optimizer.zero_grad()
            logits = model(frames)           
            loss    = criterion(logits, labels)
            loss.backward()                   
            optimizer.step()

            # update running loss
            running_loss += loss.item() * frames.size(0)
        
        print(f'  Loss: {running_loss:0.4f}')

        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device)
        val_metrics = evaluate(valloader, model, device)

        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")  

        

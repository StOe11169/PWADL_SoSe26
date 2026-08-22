import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary
from tqdm import tqdm  
from torch.utils.tensorboard import SummaryWriter
from src.evaluation import evaluate
from src.utils import get_writer, plot_confusion_matrix, build_optimizer, build_scheduler

#check Logs folder is there
os.makedirs("logs", exist_ok=True)

def trainer(trainloader, valloader, model, device, trial_number, study_dir, cfg, trial= None, writer = None, input_key="frames", pruning_step_offset=0):
    
    epochs = cfg["epochs"]
    best_f1 = -1 # -1 so best model is saved at least once, even if it does not improve F1 score
    best_epoch = 0
    best_model_path = os.path.join(study_dir, f"best_model_trial_{trial_number}.pth")
    checkpoint_path = os.path.join(study_dir, f"checkpoint_trial_{trial_number}.pth")

    #initialize Writer for tensorboard logging
    writer = get_writer(study_dir, trial_number)

    # objective function is binary cross entropy loss with logits 
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["pos_weight"], device=device)) #missclafiying yawns is twice as costly 
    

    #Removed as freezing backbone lead to terrible results early on, keeping comments as note
    # set non-trainable parameters
    #if freeze_backbone:
       # for p in model.feature_extractor.parameters():
           # p.requires_grad=False

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    # train loop
    for epoch in range(epochs): 

        # init running loss
        running_loss = 0

        # go through all data
        model.train()
        for batch in tqdm(trainloader, desc=f"Epoch {epoch}"):
            #shift data to device
            inputs = batch[input_key].to(device)
            labels = batch["labels"].to(device)


            # forward + backward pass
            optimizer.zero_grad()
            logits = model(inputs)          
            loss   = criterion(logits, labels)
            loss.backward()        
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping                
            optimizer.step()

            # update running loss
            running_loss += loss.item()
            writer.flush()

        if scheduler is not None:   
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Loss: {running_loss:0.4f}", f"    LR: {current_lr}")

        # evaluate train and validation data
        train_metrics = evaluate(trainloader, model, device, criterion, input_key= input_key)
        val_metrics = evaluate(valloader, model, device, criterion, input_key=input_key)

        #optuna pruning per epoch
        if trial is not None:
            #keep pruning steps unique across folds
            pruning_step = pruning_step_offset + epoch
            trial.report(val_metrics["f1"], pruning_step)

            if trial.should_prune():
                if writer:
                     writer.add_text("status", f"Pruned at epoch {epoch}")
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        #Log to Tensorboard
        if writer:
            writer.add_scalar("Loss/train", running_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("F1/train", train_metrics['f1'], epoch)
            writer.add_scalar("F1/val", val_metrics['f1'], epoch)
            writer.add_scalar("Precision/val", val_metrics["precision"], epoch) # high precision -> fewer false positive
            writer.add_scalar("Recall/val", val_metrics["recall"], epoch) # high recal -> more true positives

            writer.add_text("hparams", str(cfg))
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            writer.add_text("status", f"Completed (best_f1={best_f1:.4f}, epoch={best_epoch})", global_step=0)

        print(f"Train Acc: {train_metrics['accuracy']:.3f}   --   Val Acc: {val_metrics['accuracy']:.3f}")
        print(f"Train F1: {train_metrics['f1']:.3f}   --   Val F1c: {val_metrics['f1']:.3f}")  
        

        # Save best model weights and hyperparameters
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(), 'f1': best_f1, 'epoch': epoch, 'cfg': cfg, 'trial_number': trial_number}, best_model_path)

            #Save Confusion Matrix only for best model
            fig = plot_confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"], title=f"Confusion Matrix (Epoch {epoch})")
            if writer:
                writer.add_figure("Confusion_Matrix/val", fig, epoch)
            plt.close(fig)
            
        #Save best Checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': running_loss,
                     'cfg': cfg, 'trial_number': trial_number}, checkpoint_path)
            
    #Close Tensorboard writer
    if writer:
        writer.close()

    return best_f1, best_epoch
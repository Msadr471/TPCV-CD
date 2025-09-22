import argparse
import os
import shutil
import dataset.dataset as dtset
import torch
import gc
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from focal_loss.focal_loss import FocalLoss
from FAdam.fadam import FAdam

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameter for data analysis and model training.")
    
    parser.add_argument("--datapath", type=str, default='/content/Data/Mashhad/',
                        help="Path to the dataset directory")
    parser.add_argument("--log-path", type=str, default='/content/chekpoint/',
                        help="Path to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=101,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.005,
                        help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["fadam", "adamw"],
                        help="Optimizer to use: fadam or adamw")
    parser.add_argument("--loss-function", type=str, default="bce", choices=["focal", "bce"],
                        help="Loss function: focal loss or binary cross entropy")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2,
                        help="Gamma parameter for focal loss")
    parser.add_argument('--gpu-id', type=int, default=0,
                        help="GPU ID to use (if multiple GPUs available)")

    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(args.datapath))
    base_run_name = f"DT_{dataset_name}_LF_{args.loss_function}_OP_{args.optimizer}_EP_{args.epochs}_BS_{args.batch_size}_LR_{args.learning_rate}"
    
    dir_run = sorted([f for f in os.listdir(args.log_path) if f.startswith(base_run_name + "_")])
    num_run = int(dir_run[-1].split("_")[-1]) + 1 if dir_run else 0
    
    args.log_path = os.path.join(args.log_path, f"{base_run_name}_{num_run:04d}/")
    os.makedirs(args.log_path, exist_ok=True)

    return args

def create_criterion(args, loss_type='bce'):
    if loss_type == 'focal':
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean') 
    elif loss_type == 'bce':
        return torch.nn.BCELoss()
    raise ValueError(f"Unknown loss function type: {loss_type}")

def create_optimizer(model, args, optimizer_type='adamw'):
    if optimizer_type == 'fadam':
        return FAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                    betas=(0.9, 0.999), clip=1, p=0.5, eps=1.e-8)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def evaluate(model, criterion, tool4metric, device, reference, testimg, mask):
    # Reverted to training.py approach: convert to float only when needed
    reference = reference.to(device).float()
    testimg = testimg.to(device).float()
    mask = mask.to(device).float()

    # Evaluating the model:
    generated_mask = model(reference, testimg).squeeze(1)

    # Loss calculation:
    it_loss = criterion(generated_mask, mask)

    # Feeding the comparison metric tool:
    bin_genmask = (generated_mask.to("cpu") > 0.5).detach().numpy().astype(int)
    mask_np = mask.to("cpu").numpy().astype(int)
    tool4metric.update_cm(pr=bin_genmask, gt=mask_np)

    return it_loss

def log_metrics(phase, epoch, loss, scores):
    print(f"{phase} phase summary")
    print(f"Loss for epoch {epoch} is {loss}")
    
    # Extract the raw scores dictionary
    scores_dict = scores['raw_dict'] if 'raw_dict' in scores else scores
    
    # Safely get metrics with default values of 0.0 if missing
    metrics = {
        "Precision": scores_dict.get("precision_1", 0.0),
        "Recall": scores_dict.get("recall_1", 0.0),
        "OA": scores_dict["acc"],  # Overall accuracy should always be present
        "IoU": scores_dict.get("iou_1", 0.0),
        "F1": scores_dict.get("F1_1", 0.0)
    }
    
    for name, value in metrics.items():
        print(f"{name} for epoch {epoch} is {value:.4f}")
    
    print()

def training_phase(epc, model, criterion, optimizer, dataset, device, tool4metric):
    tool4metric.clear()
    model.train()
    epoch_loss = 0.0
    
    for (reference, testimg), mask in dataset:
        # Reset the gradients:
        optimizer.zero_grad()

        # Loss gradient descend step:
        it_loss = evaluate(model, criterion, tool4metric, device, reference, testimg, mask)
        it_loss.backward()  # Direct backward call like training.py
        optimizer.step()

        # Track metrics:
        epoch_loss += it_loss.to("cpu").detach().numpy()

    return epoch_loss / len(dataset)

def validation_phase(epc, model, criterion, dataset, device, tool4metric):
    model.eval()
    epoch_loss = 0.0
    tool4metric.clear()
    
    with torch.no_grad():
        for (reference, testimg), mask in dataset:
            it_loss = evaluate(model, criterion, tool4metric, device, reference, testimg, mask)
            epoch_loss += it_loss.to("cpu").detach().numpy()

    return epoch_loss / len(dataset)

def train(dataset_train, dataset_val, model, criterion, optimizer, scheduler, logpath, 
          epochs, save_after, device, training_args=None, dataset_name=None):
    
    model = model.to(device)
    tool4metric = ConfuseMatrixMeter(n_class=2)
    
    for epc in range(epochs):
        print(f"Epoch {epc}")
        
        # Training phase
        train_loss = training_phase(epc, model, criterion, optimizer, dataset_train, device, tool4metric)
        scores = tool4metric.get_scores()
        log_metrics("Training", epc, train_loss, scores)
        
        # Validation phase
        val_loss = validation_phase(epc, model, criterion, dataset_val, device, tool4metric)
        val_scores = tool4metric.get_scores()
        log_metrics("Validation", epc, val_loss, val_scores)
        
        # Save checkpoint
        if epc % save_after == 0:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'epoch': epc,
            }
            torch.save(checkpoint_data, os.path.join(logpath, f"checkpoint_{epc:03d}.pth"))
        
        scheduler.step()

def run():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    args = parse_arguments()
    
    # Load datasets
    train_data = dtset.MyDataset(args.datapath, "train")
    val_data = dtset.MyDataset(args.datapath, "val")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Current Device: {device}\n')
    
    # Clear cache before training (optional, can be removed if causing issues)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = Model()
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    criterion = create_criterion(args, args.loss_function)
    optimizer = create_optimizer(model, args, args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Start training
    train(train_loader, val_loader, model, criterion, optimizer, scheduler, args.log_path, 
          args.epochs, 1, device, vars(args), 
          os.path.basename(os.path.normpath(args.datapath)))
    

if __name__ == "__main__":
    run()
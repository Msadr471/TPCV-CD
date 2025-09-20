import argparse
import os
import shutil
import dataset.dataset as dtset
import torch
import gc
import numpy as np
import random
import subprocess
import time
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from focal_loss.focal_loss import FocalLoss
from FAdam.fadam import FAdam
from datetime import datetime
from tensorboardX import SummaryWriter

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameter for data analysis and model training.")
    
    parser.add_argument("--datapath", type=str, default='/content/Data/Mashhad/',
                        help="Path to the dataset directory")
    parser.add_argument("--log-path", type=str, default='/content/chekpoint/',
                        help="Path to save checkpoints and logs")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch number to start training from (when resuming)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=20,
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
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", 
                        choices=["efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"],
                        help="Backbone architecture for the model")
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
    reference, testimg, mask = reference.to(device).float(), testimg.to(device).float(), mask.to(device).float()
    generated_mask = model(reference, testimg).squeeze(1)
    
    bin_genmask = (generated_mask.to("cpu") > 0.5).detach().numpy().astype(int)
    mask_np = mask.to("cpu").numpy().astype(int)
    tool4metric.update_cm(pr=bin_genmask, gt=mask_np)
    
    return criterion(generated_mask, mask)

def log_metrics(phase, epoch, loss, scores, writer):
    print(f"{phase} phase summary")
    print(f"Loss for epoch {epoch} is {loss}")
    
    # Extract the raw scores dictionary
    scores_dict = scores['raw_dict'] if 'raw_dict' in scores else scores
    
    # Safely get metrics with default values of 0.0 if missing
    metrics = {
        "Loss": loss,
        "Precision": scores_dict.get("precision_1", 0.0),
        "Recall": scores_dict.get("recall_1", 0.0),
        "OA": scores_dict["acc"],  # Overall accuracy should always be present
        "IoU": scores_dict.get("iou_1", 0.0),
        "F1": scores_dict.get("F1_1", 0.0)
    }
    
    for name, value in metrics.items():
        writer.add_scalar(f"{name}_{phase}/epoch", value, epoch)
        if name != "Loss":
            print(f"{name} for epoch {epoch} is {value:.4f}")
    
    print()

def train_epoch(model, criterion, optimizer, dataset, device, tool4metric, is_training=True):
    model.train() if is_training else model.eval()
    epoch_loss = 0.0
    tool4metric.clear()
    
    for (reference, testimg), mask in dataset:
        if is_training:
            optimizer.zero_grad()
        
        loss = evaluate(model, criterion, tool4metric, device, reference, testimg, mask)
        
        if is_training:
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.to("cpu").detach().numpy()
    
    return epoch_loss / len(dataset)

def train(dataset_train, dataset_val, model, criterion, optimizer, scheduler, logpath, writer, 
          epochs, save_after, device, start_epoch=0, training_args=None, dataset_name=None):
    
    model = model.to(device)
    tool4metric = ConfuseMatrixMeter(n_class=2)
    start_time = time.time()
    
    best_metrics = {
        'train_loss': float('inf'),
        'val_loss': float('inf'),
        'f1': 0,
        'iou': 0,
        'accuracy': 0
    }
    
    for epc in range(start_epoch, epochs):
        print(f"Epoch {epc}")
        
        # Training phase
        train_loss = train_epoch(model, criterion, optimizer, dataset_train, device, tool4metric, is_training=True)
        scores = tool4metric.get_scores()
        log_metrics("Train", epc, train_loss, scores, writer)
        
        if train_loss < best_metrics['train_loss']:
            best_metrics['train_loss'] = train_loss
        
        # Validation phase
        val_loss = train_epoch(model, criterion, optimizer, dataset_val, device, tool4metric, is_training=False)
        val_scores = tool4metric.get_scores()
        log_metrics("Validation", epc, val_loss, val_scores, writer)
        
        # Update best metrics
        val_scores_dict = val_scores['raw_dict'] if 'raw_dict' in val_scores else val_scores
        for metric, value in [('val_loss', val_loss), 
                            ('f1', val_scores_dict.get("F1_1", 0.0)), 
                            ('iou', val_scores_dict.get("iou_1", 0.0)), 
                            ('accuracy', val_scores_dict["acc"])]:
            if (metric == 'val_loss' and value < best_metrics[metric]) or \
               (metric != 'val_loss' and value > best_metrics[metric]):
                best_metrics[metric] = value
        
        # Save checkpoint
        if epc % save_after == 0:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'epoch': epc,
                'training_args': training_args,
                'learning_rate': scheduler.get_last_lr()[0],
                'best_metrics': best_metrics,
                'git_hash': get_git_hash(),
                'timestamp': datetime.now().isoformat(),
                'training_time': time.time() - start_time,
                'dataset_name': dataset_name,
            }
            torch.save(checkpoint_data, os.path.join(logpath, f"checkpoint_{epc:03d}.pth"))
        
        scheduler.step()

def print_model_memory_usage(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"{name}: {param.numel():,} parameters")
    print(f"Total trainable parameters: {total_params:,}")

def run():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    args = parse_arguments()
    writer = SummaryWriter(log_dir=args.log_path)
    
    # Load datasets
    train_data = dtset.MyDataset(args.datapath, "train")
    val_data = dtset.MyDataset(args.datapath, "val")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Current Device: {device}\n')
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Use memory-efficient practices
    torch.backends.cudnn.benchmark = True  # Optimizes convolution algorithms
    
    model = Model(bkbn_name=args.backbone)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Call this after model initialization
    # print_model_memory_usage(model)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    criterion = create_criterion(args, args.loss_function)
    optimizer = create_optimizer(model, args, args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, weights_only=True)
        except:
            checkpoint = torch.load(args.resume_from, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except:
                print("Warning: Could not load optimizer state.")
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Could not load scheduler state.")
        
        if 'learning_rate' in checkpoint:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['learning_rate']
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Copy configurations
    for folder in ["models", "FAdam", "focal_loss", "dataset", "metrics"]:
        shutil.copytree(f"./{folder}", os.path.join(args.log_path, folder), dirs_exist_ok=True)
    
    # Start training
    train(train_loader, val_loader, model, criterion, optimizer, scheduler, args.log_path, 
          writer, args.epochs, 1, device, start_epoch, vars(args), 
          os.path.basename(os.path.normpath(args.datapath)))
    
    writer.close()

if __name__ == "__main__":
    run()
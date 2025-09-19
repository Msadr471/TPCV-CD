import torch
from dataset.dataset import MyDataset
import tqdm
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
import argparse
from focal_loss.focal_loss import FocalLoss
import numpy as np
import os
import random
from tqdm import tqdm
from models.change_classifier import ChangeClassifier as Model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameter for model testing.")
    
    parser.add_argument("--datapath", type=str, default='/content/Data/Mashhad/',
                        help="Path to the dataset directory")
    parser.add_argument("--modelpath", type=str, default='/content/Data/chekpoint/',
                        help="Path to the model checkpoint file")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", 
                        choices=["efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"],
                        help="Backbone architecture for the model")
    parser.add_argument("--dataset-type", type=str, default="test", 
                        choices=["val", "test"],
                        help="Dataset type to use for testing")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for testing")
    parser.add_argument("--loss-function", type=str, default="bce", 
                        choices=["focal", "bce"],
                        help="Loss function to use for evaluation")
    parser.add_argument("--focal-alpha", type=float, default=2.5e-1,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2.e+0,
                        help="Gamma parameter for focal loss")
    parser.add_argument('--gpu-id', type=int, default=0,
                        help="GPU ID to use (if multiple GPUs available)")

    args = parser.parse_args()
    return args

def create_criterion(args, loss_type='bce'):
    if loss_type == 'focal':
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean') 
    elif loss_type == 'bce':
        return torch.nn.BCELoss()
    raise ValueError(f"Unknown loss function type: {loss_type}")

def evaluate(model, criterion, tool4metric, device, reference, testimg, mask):
    reference, testimg, mask = reference.to(device).float(), testimg.to(device).float(), mask.to(device).float()
    
    with torch.no_grad():
        generated_mask = model(reference, testimg).squeeze(1)
        loss = criterion(generated_mask, mask)
        
        bin_genmask = (generated_mask.to("cpu") > 5.e-1).detach().numpy().astype(int)
        mask_np = mask.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=bin_genmask, gt=mask_np)
    
    return loss

def test_epoch(model, criterion, dataset, device, tool4metric):
    model.eval()
    epoch_loss = 0.0
    tool4metric.clear()
    
    for (reference, testimg), mask in tqdm(dataset, desc="Testing"):
        loss = evaluate(model, criterion, tool4metric, device, reference, testimg, mask)
        epoch_loss += loss.to("cpu").detach().numpy()
    
    return epoch_loss / len(dataset)

def print_results(args, loss, scores):
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Dataset path: {args.datapath}")
    print(f"Model path: {args.modelpath}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Loss function: {args.loss_function.upper()}")
    if args.loss_function == 'focal':
        print(f"Focal loss alpha: {args.focal_alpha}, gamma: {args.focal_gamma}")
    print(f"Final loss: {loss:.6f}")
    print("-"*60)
    
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
        print(f"{name}: {value:.4f}")
    
    print("="*60 + "\n")

def run():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    args = parse_arguments()
    
    # Load dataset
    dataset = MyDataset(args.datapath, args.dataset_type)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Current Device: {device}')
    
    # Initialize model
    model = Model(bkbn_name=args.backbone)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {param_count}")
    
    # Load checkpoint
    print(f"Loading model from: {args.modelpath}")
    try:
        checkpoint = torch.load(args.modelpath, weights_only=True)
    except:
        checkpoint = torch.load(args.modelpath, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Initialize criterion and metric tool
    criterion = create_criterion(args, args.loss_function)
    tool4metric = ConfuseMatrixMeter(n_class=2)
    
    # Run testing
    print(f"\nStarting testing on {args.dataset_type} dataset...")
    test_loss = test_epoch(model, criterion, test_loader, device, tool4metric)
    
    # Get scores and print results
    scores = tool4metric.get_scores()
    print_results(args, test_loss, scores)

if __name__ == "__main__":
    run()
import torch
from dataset.dataset import MyDataset
import tqdm
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier
import argparse
from focal_loss.focal_loss import FocalLoss
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/content/Data/Dataset",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="model path",
        default="/content/Data/chekpoint",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"],
        help="Backbone network used in the model"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset type to use for testing (val or test)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss function to use (bce or focal)"
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.90,
        help="Alpha parameter for Focal Loss (if used)"
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=4.0,
        help="Gamma parameter for Focal Loss (if used)"
    )

    parsed_arguments = parser.parse_args()
    
    return parsed_arguments

if __name__ == "__main__":
    args = parse_arguments()

    tool_metric = ConfuseMatrixMeter(n_class=2)

    dataset = MyDataset(args.datapath, args.dataset_type)
    test_loader = DataLoader(dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChangeClassifier(bkbn_name=args.backbone)

    # Add safe globals for numpy scalars
    with torch.serialization.safe_globals([np._core.multiarray.scalar]):
        ckpt = torch.load(args.modelpath, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]) 
    
    model.to(device)
    model.eval()

    param_tot = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of model parameters {param_tot}\n")

    loss = 0.0
    
    # Select loss function based on argument
    if args.loss_function == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
        print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    else:
        criterion = torch.nn.BCELoss()
        print("Using BCE Loss")

    with torch.no_grad():
        for (reference, testimg), mask in tqdm.tqdm(test_loader):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()
            mask = mask.float()

            generated_mask = model(reference, testimg).squeeze(1)
            
            generated_mask = generated_mask.to("cpu")
            loss += criterion(generated_mask, mask)

            bin_genmask = (generated_mask > 0.5).numpy()
            bin_genmask = bin_genmask.astype(int)
            mask = mask.numpy()
            mask = mask.astype(int)
            tool_metric.update_cm(pr=bin_genmask, gt=mask)

        loss /= len(test_loader)
        print("Test summary")
        print(f"Dataset: {args.dataset_type}")
        print(f"Batch size: {args.batch_size}")
        print(f"Loss function: {args.loss_function.upper()}")
        print("Loss is {}".format(loss))
        print()

        scores_result = tool_metric.get_scores()
        print(scores_result['formatted_output'])
        print("\nRaw metrics (for reference):")
        for key, value in scores_result.items():
            print(f"  {key}: {value}")
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
    )

    parsed_arguments = parser.parse_args()
    
    return parsed_arguments

if __name__ == "__main__":
    args = parse_arguments()

    tool_metric = ConfuseMatrixMeter(n_class=2)

    dataset = MyDataset(args.datapath, "test")
    test_loader = DataLoader(dataset, batch_size=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChangeClassifier()

    # Add safe globals for numpy scalars
    with torch.serialization.safe_globals([np._core.multiarray.scalar]):
        ckpt = torch.load(args.modelpath, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"]) 
    
    model.to(device)
    model.eval()

    param_tot = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of model parameters {param_tot}\n")

    loss = 0.0
    criterion = torch.nn.BCELoss()
    # criterion = FocalLoss(alpha=0.90, gamma=4.0, reduction='mean')

    with torch.no_grad():
        for (reference, testimg), mask in tqdm.tqdm(test_loader):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()
            mask = mask.float()

            generated_mask = model(reference, testimg).squeeze(1)
            
            generated_mask = generated_mask.to("cpu")
            loss += criterion(generated_mask, mask)

            bin_genmask = (generated_mask > 0.4).numpy()
            bin_genmask = bin_genmask.astype(int)
            mask = mask.numpy()
            mask = mask.astype(int)
            tool_metric.update_cm(pr=bin_genmask, gt=mask)

        loss /= len(test_loader)
        print("Test summary")
        print("Loss is {}".format(loss))
        print()

        scores_dictionary = tool_metric.get_scores()
        print(scores_dictionary)
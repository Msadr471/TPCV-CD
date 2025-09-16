import argparse
import os
import shutil
import dataset.dataset as dtset
import torch
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from focal_loss.focal_loss import FocalLoss
from torch.serialization import add_safe_globals
from FAdam.fadam import FAdam

def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="data path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="log path",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch to start from (when resuming)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total number of epochs to train"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.005,
        help="Weight decay"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="fadam",
        choices=["fadam", "adamw"],
        help="Optimizer to use (fadam or adamw)"
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="focal",
        choices=["focal", "bce"],
        help="Loss function to use (focal or bce)"
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.makedirs(parsed_arguments.log_path, exist_ok=True)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )
    
    # Create the run directory
    os.makedirs(parsed_arguments.log_path, exist_ok=True)

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    criterion,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    save_after,
    device,
    start_epoch=0
):

    model = model.to(device)

    tool4metric = ConfuseMatrixMeter(n_class=2)

    # Remove the debug prints from evaluate function:
    def evaluate(reference, testimg, mask):
        # All the tensors on the device:
        reference = reference.to(device).float()
        testimg = testimg.to(device).float()
        mask = mask.to(device).float()

        # Evaluating the model:
        generated_mask = model(reference, testimg)
        
        # Remove channel dimension from model output:
        generated_mask = generated_mask.squeeze(1)

        # Loss gradient descend step:
        it_loss = criterion(generated_mask, mask)

        # Feeding the comparison metric tool:
        bin_genmask = (generated_mask.to("cpu") > 0.5).detach().numpy().astype(int)
        mask_np = mask.to("cpu").numpy().astype(int)
        tool4metric.update_cm(pr=bin_genmask, gt=mask_np)

        return it_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        for (reference, testimg), mask in dataset_train:
            # Reset the gradients:
            optimizer.zero_grad()

            # Loss gradient descend step:
            it_loss = evaluate(reference, testimg, mask)
            it_loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("Precision/epoch", 
                          scores_dictionary["precision_1"], epc)
        writer.add_scalar("Recall/epoch", 
                          scores_dictionary["recall_1"], epc)
        writer.add_scalar("OA/epoch", 
                          scores_dictionary["acc"], epc)
        writer.add_scalar("IoU class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "Precision for epoch {} is {:.4f}".format(
                epc, scores_dictionary["precision_1"]
            )
        )
        print(
            "Recall for epoch {} is {:.4f}".format(
                epc, scores_dictionary["recall_1"]
            )
        )
        print(
            "Overal Accuracy for epoch {} is {:.4f}".format(
                epc, scores_dictionary["acc"]
            )
        )
        print(
            "IoU for epoch {} is {:.4f}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 for epoch {} is {:.4f}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()
        writer.flush()

        ### Save the model ###
        if epc % save_after == 0:
            # Format epoch number with leading zeros (3 digits)
            epoch_str = str(epc).zfill(3)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'epoch': epc,
            }, os.path.join(logpath, f"checkpoint_{epoch_str}.pth"))

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        with torch.no_grad():
            for (reference, testimg), mask in dataset_val:
                epoch_loss_eval += evaluate(reference,
                                            testimg, mask).to("cpu").numpy()

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        writer.add_scalar("Precision_val/epoch", 
                          scores_dictionary["precision_1"], epc)
        writer.add_scalar("Recall_val/epoch", 
                          scores_dictionary["recall_1"], epc)
        writer.add_scalar("OA_val/epoch", 
                          scores_dictionary["acc"], epc)
        writer.add_scalar("IoU_val class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "Precision for epoch {} is {:.4f}".format(
                epc, scores_dictionary["precision_1"]
            )
        )
        print(
            "Recall for epoch {} is {:.4f}".format(
                epc, scores_dictionary["recall_1"]
            )
        )
        print(
            "Overal Accuracy for epoch {} is {:.4f}".format(
                epc, scores_dictionary["acc"]
            )
        )
        print(
            "IoU for epoch {} is {:.4f}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 for epoch {} is {:.4f}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()

    for epc in range(start_epoch, epochs):
        training_phase(epc)
        validation_phase(epc)
        # scheduler step
        scheduler.step()


def run():

    # set the random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)

    # Inizialitazion of dataset and dataloader:
    trainingdata = dtset.MyDataset(args.datapath, "train")
    validationdata = dtset.MyDataset(args.datapath, "val")
    data_loader_training = DataLoader(trainingdata, batch_size=args.batch_size, shuffle=True)
    data_loader_val = DataLoader(validationdata, batch_size=args.batch_size, shuffle=True)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model()
    start_epoch = args.start_epoch

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    def create_criterion(loss_type='focal', **kwargs):
        if loss_type == 'focal':
            return FocalLoss(alpha=0.90, gamma=4.0, reduction='mean')
        elif loss_type == 'bce':
            return torch.nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    # define the loss function for the model training.
    criterion = create_criterion(loss_type=args.loss_function)
    
    # choose the optimizer with configurable parameters
    def create_optimizer(model, args, optimizer_type='fadam'):
        if optimizer_type == 'fadam':
            return FAdam(
                model.parameters(), 
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999),
                clip=1.0,
                p=0.5,
                eps=1e-8,
                momentum_dtype=torch.float32,
                fim_dtype=torch.float32,
                maximize=False
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                model.parameters(), 
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=False
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Then use it:
    optimizer = create_optimizer(model, args, optimizer_type=args.optimizer)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # RESUME TRAINING LOGIC
    # In the resume training section, you might need to adjust this:
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, weights_only=True)
        except:
            print("Secure loading failed, falling back to insecure mode")
            checkpoint = torch.load(args.resume_from, weights_only=False)
        
        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Only load optimizer state if it's compatible (same optimizer type)
        # You might want to skip loading optimizer state when switching optimizers
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer tensors to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except:
                print("Warning: Could not load optimizer state. Starting with fresh optimizer.")
        
        # Load scheduler if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # copy the configurations
    os.makedirs(os.path.join(args.log_path, "models"), exist_ok=True)
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
        dirs_exist_ok=True  # Added for Python 3.8+ compatibility
    )
    _ = shutil.copytree(
        "./FAdam",  # or the correct path to your FAdam folder
        os.path.join(args.log_path, "FAdam"),
        dirs_exist_ok=True
    )
    _ = shutil.copytree(
        "./focal_loss",  # Add this line
        os.path.join(args.log_path, "focal_loss"),
        dirs_exist_ok=True
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=args.epochs,
        save_after=1,
        device=device,
        start_epoch=start_epoch
    )
    writer.close()


if __name__ == "__main__":
    run()
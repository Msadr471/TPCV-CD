import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset import MyDataset
from models.change_classifier import ChangeClassifier
from torch.utils.data import DataLoader, Subset
import argparse
import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Loss Landscape Plotting")
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--radius", type=float, default=1)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=160)
    return parser.parse_args()


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def unflatten_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx + numel].view_as(p))
        idx += numel


def compute_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (ref, test), mask in loader:
            ref = ref.to(device).float()
            test = test.to(device).float()
            mask = mask.to(device).float()
            out = model(ref, test).squeeze(1)
            loss = criterion(out, mask)
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_loss_landscape(model, loader, criterion, device, steps, radius):
    base_params = flatten_params(model).detach().cpu()

    np.random.seed(42)
    dir1 = torch.randn_like(base_params)
    dir1 /= torch.norm(dir1)
    dir2 = torch.randn_like(base_params)
    dir2 -= torch.dot(dir2, dir1) * dir1
    dir2 /= torch.norm(dir2)

    losses = np.zeros((steps, steps))
    alphas = np.linspace(-radius, radius, steps)
    betas = np.linspace(-radius, radius, steps)

    for i, alpha in enumerate(tqdm.tqdm(alphas, desc="Computing loss landscape")):
        for j, beta in enumerate(betas):
            perturbed = base_params + alpha * dir1 + beta * dir2
            unflatten_params(model, perturbed.to(device))
            loss = compute_loss(model, loader, criterion, device)
            losses[i, j] = loss

    unflatten_params(model, base_params.to(device))  # restore

    X, Y = np.meshgrid(alphas, betas)

    # Save results safely on CPU
    np.savez("loss_landscape_data.npz", losses=losses, alphas=alphas, betas=betas)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, losses, cmap="magma", edgecolor="k", linewidth=0.3, antialiased=True)

    ax.set_title("Loss Landscape")
    ax.set_xlabel("Direction 1 (α)")
    ax.set_ylabel("Direction 2 (β)")
    ax.set_zlabel("Loss")

    plt.tight_layout()
    plt.savefig("loss_landscape_3d.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, losses, levels=50, cmap='magma')
    plt.colorbar(cp)
    plt.title("Loss Landscape Contours")
    plt.xlabel("Direction 1 (α)")
    plt.ylabel("Direction 2 (β)")
    plt.tight_layout()
    plt.savefig("loss_landscape_contours.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()

    dataset = MyDataset(args.datapath, "val")

    if args.subset_size and args.subset_size < len(dataset):
        indices = list(range(args.subset_size))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = ChangeClassifier()
    model.load_state_dict(torch.load(args.modelpath, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Using device: {device}")

    criterion = torch.nn.BCELoss()

    plot_loss_landscape(model, loader, criterion, device, args.steps, args.radius)

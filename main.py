import argparse
import torch
from train import train
from sampler import sample
from utils import load_model
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_samples(model_path, num_samples=16, img_size=28, max_timestep=1000):
    model = load_model(model_path, device=device)
    samples = sample(model, num_samples=num_samples, img_size=img_size, max_timestep=max_timestep, device=device)
    
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(num_samples):
        axs[i // 4, i % 4].imshow(samples[i, 0].cpu().numpy(), cmap="gray")
        axs[i // 4, i % 4].axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM MNIST Runner")
    subparsers = parser.add_subparsers(dest="command")

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a DDPM model")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--model_name", type=str, default="models/model.pt")
    train_parser.add_argument("--max_timestep", type=int, default=1000)

    # SAMPLE
    sample_parser = subparsers.add_parser("sample", help="Generate samples from a trained model")
    sample_parser.add_argument("--model_path", type=str, required=True)
    sample_parser.add_argument("--num_samples", type=int, default=16)
    sample_parser.add_argument("--img_size", type=int, default=28)
    sample_parser.add_argument("--max_timestep", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "train":
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_name=args.model_name,
            max_timestep=args.max_timestep,
            device=device
        )
    elif args.command == "sample":
        generate_samples(
            model_path=args.model_path,
            num_samples=args.num_samples,
            img_size=args.img_size,
            max_timestep=args.max_timestep
        )
    else:
        parser.print_help()
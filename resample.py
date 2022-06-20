import rich
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import imageio
import cv2
import typer
from pathlib import Path


def resample(
    checkpoint: str = typer.Option(..., "-c", "--checkpoint"),
    filename: str = typer.Option(..., "-f", "--filename"),
    timesteps: int = typer.Option(100, "-t", "--timesteps"),
    q_posterior: bool = typer.Option(False, "--q/--noq"),
):

    rich.print(f"Denoise configuration: {timesteps} {q_posterior}")
    filename = Path(filename)

    # MODEL
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()
    image_size = [256, 256 * 4]
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        loss_type="l1",  # number of steps  # L1 or L2
    ).cuda()

    # Reload Weights
    state_dict = torch.load(checkpoint)
    diffusion.load_state_dict(state_dict["model"])

    # img = imageio.imread("pino.png")
    img = imageio.imread(filename)
    img = cv2.resize(img, (image_size[1], image_size[0]))
    print("Image shape", img.shape)

    # to tensor
    img = torch.tensor(img / 255).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1

    # Forward / Backward passes
    desampled_images, middle_images = diffusion.forward_and_back(
        img, timesteps, use_q_posterior=False
    )

    # to numpy
    deimg = desampled_images[0, ::].permute(1, 2, 0).detach().cpu().numpy()
    miimg = middle_images[0, ::].permute(1, 2, 0).detach().cpu().numpy()

    tag = f"_{timesteps}_{'Q' if q_posterior else ''}"
    resample_filaname = filename.parent / (filename.stem + f"_resample_{tag}.png")
    middle_filename = filename.parent / (filename.stem + f"_middle_{tag}.png")

    imageio.imwrite(resample_filaname, deimg)
    imageio.imwrite(middle_filename, miimg)


if __name__ == "__main__":
    typer.run(resample)

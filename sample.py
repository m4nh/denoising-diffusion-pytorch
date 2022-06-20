from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
import imageio

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()
image_size = [256, 256 * 4]
diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,
    loss_type="l1",  # number of steps  # L1 or L2
).cuda()

state_dict = torch.load(
    "/home/daniele/work/workspace_python/denoising-diffusion-pytorch/results/model-22.pt"
)
diffusion.load_state_dict(state_dict["model"])

sampled_images = diffusion.sample(batch_size=1)

img = sampled_images[0, ::].permute(1, 2, 0).detach().cpu().numpy()
imageio.imwrite("sample.png", img)
print(sampled_images.shape)

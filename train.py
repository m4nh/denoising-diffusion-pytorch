from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()
image_size = 256
diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,
    loss_type="l1",  # number of steps  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    "/home/daniele/Desktop/experiments/2022-06-14.StylishSonoco/datasets/sonoco_labelled_train/data",
    train_batch_size=16,
    train_lr=1e-4,
    image_size=image_size,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    save_and_sample_every=5000
)

trainer.train()

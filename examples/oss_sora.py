import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform



import sys
sys.path.append('../OSS')
from OSS.OSS import search_OSS_video, infer_OSS
from OSS.model_wrap import _WrappedModel_Sora



@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        teacher_steps=200,
        student_steps=20,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        
        self.teacher_steps = teacher_steps
        self.student_steps = student_steps

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )


    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        
        
        if additional_args is not None:
            model_args.update(additional_args)

        teacher_steps = self.teacher_steps
        student_steps = self.student_steps

        # prepare timesteps
        timesteps = [(1.0 - i / teacher_steps) * self.num_timesteps for i in range(teacher_steps)]
        if self.use_discrete_timesteps: #None
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        mask_t = mask * self.num_timesteps


        
        B = z.shape[0]
        context = model_args["y"]
        del model_args["y"]
        model = _WrappedModel_Sora(model, guidance_scale, y_null, timesteps, self.num_timesteps, mask_t)
        
        oss_steps = search_OSS_video(model, z, B, context, device, teacher_steps=teacher_steps, student_steps=student_steps, norm=2, model_kwargs=model_args, float32=False)
        z_oss = infer_OSS(oss_steps, model, z, context, device, float32=False, model_kwargs=model_args)

        # teacher video
        teacher_steps = list(range(1, teacher_steps + 1))
        z_tea = infer_OSS(teacher_steps, model, z, context, device, float32=False, model_kwargs=model_args)
        
        return z_oss, z_tea




    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

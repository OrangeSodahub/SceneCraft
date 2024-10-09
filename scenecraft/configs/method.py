from pathlib import Path
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from scenecraft.data.datamanager import SceneCraftDataManagerConfig
from scenecraft.data.dataset import DataParserConfig
from scenecraft.trainer import SceneCraftTrainerConfig
from scenecraft.model import SceneCraftModelConfig
from scenecraft.pipeline import SceneCraftPipelineConfig


scenecraft_method_nerfacto_big2 = MethodSpecification(
    config=SceneCraftTrainerConfig(
        method_name="nerfacto-big2",
        experiment_name="nerf",
        steps_per_eval_batch=50,
        steps_per_eval_image=50,
        steps_per_save=200,
        max_num_iterations=15000,
        save_only_latest_checkpoint=False,
        mixed_precision=True,
        gradient_accumulation_steps=1,
        pipeline=SceneCraftPipelineConfig(
            enable_modelAB=False,
            async_mode=True,
            full_image_every=1,
            diffusion_steps=20,
            scheduler_type="unipc",
            guidance_depth=False,
            fix_init_noise=False,
            controlnet_conditioning_scale=[3.5, 1.5],
            datamanager=SceneCraftDataManagerConfig(
                dataparser=DataParserConfig(
                    dataset_type="scannetpp",
                    data=Path("outputs/scannetpp"),
                    scene_id="0a7cc12c0e",
                    # dataset_type="hypersim",
                    # data=Path("outputs/hypersim"),
                    # scene_id="ai_001_005",
                    # dataset_type="custom",
                    # data=Path("outputs/custom"),
                    # scene_id="exp5",
                ),
                condition_type="one_hot",
                train_num_images_to_sample_from=2,
                train_num_times_to_repeat_images=0,
                nerf_batch_size=1,
                # camera_res_scale_factor=0.5754, # [584, 876] -> [336, 504]
                # guide_camera_res_scale_factor=0.8768,
                camera_res_scale_factor=0.4375,
                guide_camera_res_scale_factor=0.6875,
                guide_buffer_size=250,
            ),
            downscale_factor=0.5,
            prompt="This is one view of a bedroom painted by VanGogh.",
            time_schedule=[
                [600, -1, -1], [1000, 0.8, 0.98], [1300, 0.7, 0.8], [1600, 0.6, 0.7], [1900, 0.5, 0.6], [2200, 0.2, 0.5]],
            model=SceneCraftModelConfig(
                use_lpips=True,
                use_l1=False,
                use_latent_loss=True,
                lpips_loss_mult=0.1,
                style_loss_mult=20.,
                latent_loss_mult=1,
                rgb_loss_mult=5.,
                interlevel_loss_mult=.5,
                distortion_loss_mult=0.002,
                depth_loss_mult=0.5,
                guide_depth_loss_mult=0.2,
                depth_consistency_loss_mult=5.,
                zvar_loss_mult=3.,
                use_appearance_embedding=True,
                use_direction_encoding=False,
                compute_background_color=False,
                eval_num_rays_per_chunk=1 << 15, # 32768
                train_num_rays_per_chunk=1 << 15, # 32768
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=20000,
                log2_hashmap_size=21,
                patch_size=224,
            ),
            checkpoint_path="gzzyyxy/layout_diffusion_scannetpp_prompt_one_hot_multi_control_bs32_epoch18",
            checkpoint_subfolder="checkpoint-15000",
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
            guidance_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=10.),
                "scheduler": None,

            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=10.),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="",
)


scenecraft_method_nerfacto_huge2 = MethodSpecification(
    config=SceneCraftTrainerConfig(
        method_name="nerfacto-huge2",
        experiment_name="nerf",
        steps_per_eval_batch=50,
        steps_per_eval_image=50,
        steps_per_save=200,
        max_num_iterations=15000,
        save_only_latest_checkpoint=False,
        mixed_precision=True,
        gradient_accumulation_steps=1,
        pipeline=SceneCraftPipelineConfig(
            enable_modelAB=False,
            async_mode=True,
            full_image_every=1,
            diffusion_steps=20,
            scheduler_type="unipc",
            guidance_depth=False,
            fix_init_noise=False,
            controlnet_conditioning_scale=[3.5, 1.5],
            datamanager=SceneCraftDataManagerConfig(
                dataparser=DataParserConfig(
                    dataset_type="scannetpp",
                    data=Path("outputs/scannetpp"),
                    scene_id="0a7cc12c0e",
                    # dataset_type="hypersim",
                    # data=Path("outputs/hypersim"),
                    # scene_id="ai_001_005",
                    # dataset_type="custom",
                    # data=Path("outputs/custom"),
                    # scene_id="exp5",
                ),
                condition_type="one_hot",
                train_num_images_to_sample_from=2,
                train_num_times_to_repeat_images=0,
                nerf_batch_size=1,
                # camera_res_scale_factor=0.5754, # [584, 876] -> [336, 504]
                # guide_camera_res_scale_factor=0.8768,
                camera_res_scale_factor=0.4375,
                guide_camera_res_scale_factor=0.6875,
                guide_buffer_size=250,
            ),
            downscale_factor=0.5,
            prompt="This is one view of a bedroom painted by VanGogh.",
            time_schedule=[
                [600, -1, -1], [1000, 0.8, 0.98], [1300, 0.7, 0.8], [1600, 0.6, 0.7], [1900, 0.5, 0.6], [2200, 0.2, 0.5]],
            model=SceneCraftModelConfig(
                use_lpips=True,
                use_l1=False,
                use_latent_loss=True,
                lpips_loss_mult=0.1,
                style_loss_mult=20.,
                latent_loss_mult=1,
                rgb_loss_mult=5.,
                interlevel_loss_mult=.5,
                distortion_loss_mult=0.002,
                depth_loss_mult=0.5,
                guide_depth_loss_mult=0.2,
                depth_consistency_loss_mult=5.,
                zvar_loss_mult=3.,
                use_appearance_embedding=True,
                use_direction_encoding=False,
                compute_background_color=False,
                eval_num_rays_per_chunk=1 << 15, # 32768
                train_num_rays_per_chunk=1 << 15, # force to be same as the backward process
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(512, 512),
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
                ],
                hidden_dim=256,
                hidden_dim_color=256,
                appearance_embed_dim=32,
                max_res=8192,
                proposal_weights_anneal_max_num_iters=20000,
                log2_hashmap_size=21,
                patch_size=224,
            ),
            # checkpoint_path="gzzyyxy/layout_diffusion_hypersim_prompt_one_hot_multi_control_bs32_epoch24",
            # checkpoint_subfolder="checkpoint-10900",
            checkpoint_path="gzzyyxy/layout_diffusion_scannetpp_prompt_one_hot_multi_control_bs32_epoch18",
            checkpoint_subfolder="checkpoint-15000",
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
            guidance_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=10.),
                "scheduler": None,

            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, max_norm=10.),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="",
)

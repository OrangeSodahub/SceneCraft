# Adapted from https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/prompt_processors/base.py

import os
import torch
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional
from jaxtyping import Float
from nerfstudio.utils.rich_utils import CONSOLE
from transformers import AutoTokenizer, CLIPTextModel, PreTrainedTokenizer

from nerfstudio.utils import comms
from scenecraft.utils import barrier, cleanup


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class PromptProcessorOutput:
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    text_embeddings: Float[torch.Tensor, "N max_len embed_dim"] = None
    uncond_text_embeddings: Optional[Float[torch.Tensor, "N max_len embed_dim"]] = None

    def get_text_embeddings(self) -> Float[torch.Tensor, "N max_len embed_dim"]:
        if self.uncond_text_embeddings is not None:
            return torch.cat([self.uncond_text_embeddings, self.text_embeddings])
        else:
            return self.text_embeddings

    def to(self, *args, **kwargs):
        args = args + tuple(kwargs.values())
        assert len(args) <= 2, f"Maximum number of supported arguments is 2."
        if len(args) == 2:
            assert type(args[0]) != type(args[1]), f"Got two repeated arguments {args[0]}."
        device = None
        dtype = None
        for arg in args:
            if isinstance(arg, torch.device) and device is None:
                device = arg
            elif isinstance(arg, torch.dtype) and dtype is None:
                dtype = arg
        if device is not None:
            self.device = device
            self.text_embeddings = self.text_embeddings.to(device=device)
            self.uncond_text_embeddings = self.uncond_text_embeddings.to(device=device)
        if dtype is not None:
            self.dtype = dtype
            self.text_embeddings = self.text_embeddings.to(dtype=dtype)
            self.uncond_text_embeddings = self.uncond_text_embeddings.to(dtype=dtype)
        return self


class PromptProcessor(object):
    def __init__(self,
                 pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
                 tokenizer: PreTrainedTokenizer = None,
                 text_encoder: CLIPTextModel = None,
                 prompts: Union[str, List[str]] = None,
                 do_classifier_free_guidance: bool = False,
                 device: Union[torch.device, str] = "cpu",
                 use_cache: bool = True,
                 cache_dir: str = ".cache/text_embeddings",
                 spawn: bool = False, **kwargs, ) -> None:

        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.spawn = spawn
        self.do_classifier_free_guidance = do_classifier_free_guidance

        # configure tokenizer and text_encoder
        if tokenizer is not None and text_encoder is not None:
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
            CONSOLE.print(f"Loaded tokenizer and text_encoder from guidance pipeline.")
        else:
            assert pretrained_model_name_or_path is not None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="text_encoder"
            ).to(self.device)
            CONSOLE.print(f"Loaded tokenizer and text_encoder from {pretrained_model_name_or_path}.")

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # configure prompts is possible
        if prompts is not None:
            self.prepare_text_embeddings(prompts)

    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def prepare_text_embeddings(self, prompts: Union[str, List[str]]):
        os.makedirs(self.cache_dir, exist_ok=True)
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts_to_process = []
        for prompt in set(prompts):
            if self.use_cache:
                # some text embeddings are already in cache and do not process them
                cache_path = os.path.join(
                    self.cache_dir, f"{hash_prompt(self.pretrained_model_name_or_path, prompt)}.pt")
                if os.path.exists(cache_path):
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            if self.spawn:
                ctx = torch.multiprocessing.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(self.pretrained_model_name_or_path, prompts_to_process,
                          self.cache_dir, self.tokenizer, self.text_encoder)
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.pretrained_model_name_or_path, prompts_to_process, self.cache_dir,
                    self.tokenizer, self.text_encoder,
                )
            cleanup()

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self.cache_dir,
            f"{hash_prompt(self.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    def get_text_embeddings(self, *args, **kwargs) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    @staticmethod
    def spawn_func(*args, **kwargs):
        raise NotImplementedError

    def __call__(self, prompts: Union[str, List[str]], negative_prompts: Optional[Union[str, List[str]]] = None,
                 ) -> PromptProcessorOutput:
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier(comms.LOCAL_PROCESS_GROUP)
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        text_embeddings = torch.stack(list(map(self.load_from_cache, prompts))) # [B, max_len, embed_dim]
        # get the negative prompt embeddings
        uncond_text_embeddings = None
        if self.do_classifier_free_guidance:
            if negative_prompts is None:
                negative_prompts = [""] * batch_size
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts]
            self.prepare_text_embeddings(negative_prompts)
            assert len(negative_prompts) == batch_size, f"`negative_prompts` should has the same batch_size as the `prompts`."
            uncond_text_embeddings = torch.stack(list(map(self.load_from_cache, negative_prompts))) # [B, max_len, embed_dim]
        return PromptProcessorOutput(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
            text_embeddings=text_embeddings,
            uncond_text_embeddings=uncond_text_embeddings,
        )


class StableDiffusionPromptProcessor(PromptProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, tokenizer, text_encoder):

        CONSOLE.print(f"Encoding and caching {len(prompts)} prompts.")
        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

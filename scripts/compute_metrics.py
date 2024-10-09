import torch
import os
import yaml
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn
from tqdm import tqdm
from PIL import Image
from scipy import linalg
from scipy.stats import entropy
from typing import List
from pathlib import Path
from sklearn.metrics.pairwise import polynomial_kernel
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from scenecraft.data.dataset import RAW_DATASETS
from scenecraft.utils import load_from_jsonl
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
        use_fid_inception=True,
    ):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights="DEFAULT")

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split(".")[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs["init_weights"] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and "weights" in kwargs:
        if kwargs["weights"] == "DEFAULT":
            kwargs["pretrained"] = True
        elif kwargs["weights"] is None:
            kwargs["pretrained"] = False
        else:
            raise ValueError(
                "weights=={} not supported in torchvision {}".format(
                    kwargs["weights"], torchvision.__version__
                )
            )
        del kwargs["weights"]

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None, split=False):
        self.files = files
        self.split = split
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.split:
            width, height = img.size
            img = img.crop((width // 3, 0, width // 3 * 2, height))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class Metric:
    def __init__(self, device, batch_size, num_workers, **kwargs):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.inputs = []

    def _calculate_score(self):
        raise NotImplementedError

    def _run_scenecraft(self, metas: List[dict], **kwargs):
        for i in tqdm(range(len(metas)), leave=False):
            dataset, scene_id = metas[i]["dataset"], metas[i]["scene_id"]

            output_path = os.path.join("outputs", dataset, scene_id)
            if not os.path.exists(output_path):
                raise FileExistsError

            # find image path
            if "ref_image" not in self.inputs: # for nerf renderings
                save_id = metas[i]["save_id"]
                render_path = [p for p in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, p))
                                                                    and p.startswith(f"render3d_{str(save_id).zfill(2)}")]
                image_path = os.path.join(output_path, render_path[0], "rgb")
            else: # for diffusion generations
                image_path = os.path.join(output_path, "diffusion2d_real")

            if not image_path:
                return None
            inputs = (Path(image_path),)

            # get prompts
            if "prompt" in self.inputs:
                infos = load_from_jsonl(Path(output_path) / "all.jsonl")
                items = list(map(lambda info:
                                list(map(lambda label: RAW_DATASETS[dataset].label2cat[label], info["item_id"])), infos))
                if "prompt" in infos[0]:
                    prompts = list(map(lambda info: info["prompt"], infos))
                else:
                    prompts = [' '.join(render_path.split('_')[2:])] * len(infos)
                prompts = list(map(lambda prompt, items: f"{prompt.strip('.')}, which contains {' '.join(items)}.", prompts, items))
                inputs += (prompts,)

            # find ref image path
            if "ref_image" in self.inputs:
                ref_image_path = image_path
                inputs += (Path(ref_image_path),)

            metas[i]["score"] = self._calculate_score(*inputs)

        return metas

    def _run_setthescene(self, root: os.PathLike, **kwargs):
        exp_dir = os.path.join(root, "experiments")
        exps = os.listdir(exp_dir)
        print(f"Find {len(exps)} experiments: {' '.join(exps)}")

        # calulate score for each experiment
        metas = []
        for exp in exps:
            exp_path = os.path.join(exp_dir, exp)
            config_path = os.path.join(exp_path, "config.yaml")
            if not os.path.exists(config_path):
                continue
            config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
            image_path = os.path.join(exp_path, "results", "images")
            inputs = (Path(image_path),)
            if "prompt" in self.inputs:
                prompt = f"{config['guide']['text']}, with " + " ".join(config["scene_nerfs"]["nerf_texts"])
                prompts = [prompt] * 25
                inputs += (prompts,)
            score = self._calculate_score(*inputs)
            metas.append({"exp": exp, "score": score})
        return metas

    def _run_mvdiffusion(self, root: os.PathLike, **kwargs):
        outs_dir = os.path.join(root, "outputs")
        results = os.listdir(outs_dir)
        print(f"Find {len(results)} results: {' '.join(results)}")

        # calculate score for each result
        metas = []
        for result in tqdm(results):
            result_path = os.path.join(outs_dir, result)
            if len(os.listdir(result_path)) < 8:
                continue
            inputs = (Path(result_path),)

            if "prompt" in self.inputs:
                prompt_file = os.path.join(result_path, "prompt.txt")
                prompt = open(prompt_file, "r").readlines()[0]
                if "\n" in prompt:
                    prompt = prompt.strip()
                prompts = prompt.split("This is one view of")
                if len(prompts) == 1:
                    prompts = [prompt] * 8
                else:
                    if len(prompts) == 9:
                        assert prompts[0] == ''
                        prompts = prompts[1:]
                    assert len(prompts) == 8, f"Number of prompts {len(prompts)} should be 8."
                    prompts = list(map(lambda x: "This is one viewe of" + x, prompts))
                inputs += (prompts,)

            if "ref_image" in self.inputs:
                pass

            score = self._calculate_score(*inputs)
            metas.append({"result": result, "score": score})
        return metas

    def __call__(self, method: str, verbose=True, **kwargs):
        if not hasattr(self, f"_run_{method}"):
            raise NotImplementedError

        metas = getattr(self, f"_run_{method}")(**kwargs)

        if verbose:
            print("Summary:")
            scores = []
            for meta in metas:
                print(", ".join([f"{k}: {v}" for k, v in meta.items()]))
                scores.append(meta["score"])
            print(f"Mean score: {np.mean(np.array(scores))}")


class CLIPScore(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print(f"Loaded Model {self.model.__class__.__name__}")

        self.inputs = ["image", "prompt"]

    @torch.no_grad()
    def _calculate_score(self, path, texts: List[str]=None):
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        dataset = ImagePathDataset(files)
        images = [dataset[i] for i in range(len(dataset))]
        batch_size = min(len(dataset), self.batch_size)
        mean_scores = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(text=texts[i : i + batch_size], images=images[i : i + batch_size], return_tensors="pt", padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image # [N, N]
            mean_score = torch.mean(logits_per_image, dim=0)
            mean_scores.append(mean_score)
        mean_score = torch.mean(torch.concat(mean_scores)).item()
        return mean_score


class InceptionScore(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.model = _inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        print(f"Loaded Model {self.model.__class__.__name__}")

        self.inputs = ["image"]

    @torch.no_grad()
    def _calculate_score(self, path, splits=10):
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        dataset = ImagePathDataset(files,
                    transforms=transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                    ]))
        batch_size = min(len(dataset), self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        outputs = []
        for batch in tqdm(dataloader, leave=False):
            output = self.model(batch.cuda()).softmax(dim=1).data.cpu().numpy()
            outputs.append(output)
        outputs = np.concatenate(outputs) # [N, 1000]

        if outputs.shape[0] // splits < 1:
            import warnings
            warnings.warn("Number of samples less than splits=10.", UserWarning)
            splits = max(2, outputs.shape[0] // 2)

        # compute kl divergence
        split_scores = []
        for k in range(splits):
            part = outputs[k * (len(outputs) // splits) : (k + 1) * (len(outputs) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(scores))

        return np.mean(np.array(split_scores))


class KIDScore(Metric):
    def __init__(self, device, batch_size, dims, num_workers):
        super().__init__(device, batch_size, num_workers)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()
        print(f"Loaded Model {self.model.__class__.__name__}")

        self.inputs = ["image", "ref_image"]

    def _calculate_statistics(self, path, real=False):
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        if not real:
            files = list(filter(lambda fpath: "real" not in fpath.name, files))
        else:
            files = list(filter(lambda fpath: "real" in fpath.name, files))

        dataset = ImagePathDataset(files,
                    transforms=transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                    ]), split=not real)
        batch_size = min(len(dataset), self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        features = []            
        for batch in tqdm(dataloader, leave=False):
            feature = self.model(batch.cuda())[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if feature.size(2) != 1 or feature.size(3) != 1:
                feature = adaptive_avg_pool2d(feature, output_size=(1, 1))

            feature = feature.squeeze(3).squeeze(2).cpu().numpy()
            features.append(feature)

        return np.concatenate(features)

    def _calculate_score(self, path1: os.PathLike, path2: os.PathLike):
        feat1 = self._calculate_statistics(path1, real=False)
        feat2 = self._calculate_statistics(path2, real=True)
        kernel_matrix = polynomial_kernel(feat1, feat2)
        kid = np.trace(kernel_matrix) + np.trace(feat1) + np.trace(feat2) - 3 * np.trace(linalg.sqrtm(kernel_matrix))
        return kid


class FIDScore(Metric):
    def __init__(self, device, batch_size, dims, num_workers):
        super().__init__(device, batch_size, num_workers)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()
        print(f"Loaded Model {self.model.__class__.__name__}")

        self.inputs = ["image", "ref_image"]

    def _calculate_activation(self, files=None, split=False):
        dataset = ImagePathDataset(files,
                    transforms=transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                    ]), split=split)
        batch_size = min(self.batch_size, len(files))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        outputs = []
        for batch in tqdm(dataloader, leave=False):
            batch = batch.to(self.device)

            with torch.no_grad():
                output = self.model(batch.cuda())[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if output.size(2) != 1 or output.size(3) != 1:
                output = adaptive_avg_pool2d(output, output_size=(1, 1))

            output = output.squeeze(3).squeeze(2).cpu().numpy()
            outputs.append(output)

        outputs = np.concatenate(outputs)
        return outputs

    def _calculate_statistics(self, path, real=False):
        path = Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        if not real:
            files = list(filter(lambda fpath: "real" not in fpath.name, files))
        else:
            files = list(filter(lambda fpath: "real" in fpath.name, files))

        activations = self._calculate_activation(files=files, split=not real)
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def _calculate_score(self, path1: os.PathLike, path2: os.PathLike):
        m1, s1 = self._calculate_statistics(path1, real=False)
        m2, s2 = self._calculate_statistics(path2, real=True)
        fid = self._calculate_frechet_distance(m1, s1, m2, s2)
        return fid

    def save_stats(self, path):
        m1, s1 = self._calculate_statistics(path)
        np.savez_compressed(path, mu=m1, sigma=s1)


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.metric.upper() == "CS":
        Metric = CLIPScore(device, args.batch_size, num_workers)
    elif args.metric.upper() == "IS":
        Metric = InceptionScore(device, args.batch_size, num_workers)
    elif args.metric.upper() == "FID":
        Metric = FIDScore(device, args.batch_size, args.dims, num_workers)
    elif args.metric.upper() == "KID":
        Metric = KIDScore(device, args.batch_size, args.dims, num_workers)
    else:
        raise ValueError(f"Unknown metric type {args.metric}.")

    if args.save_stats:
        Metric.save_stats(args.path)
        return

    metric = Metric(args.method, root=args.root, metas=metas)
    print(f"Calculated {args.metric}: {metric}")


################################ CLIP SCORE, INCEPTION SCORE ##########################
metas = [
    {"dataset": "hypersim", "scene_id": "ai_001_006", "save_id": 0},
    {"dataset": "hypersim", "scene_id": "ai_001_008", "save_id": 0},
    {"dataset": "hypersim", "scene_id": "ai_010_004", "save_id": 0},
    {"dataset": "hypersim", "scene_id": "ai_010_004", "save_id": 1},
    {"dataset": "hypersim", "scene_id": "ai_010_004", "save_id": 2},
    {"dataset": "hypersim", "scene_id": "ai_010_005", "save_id": 0},
    {"dataset": "hypersim", "scene_id": "ai_012_005", "save_id": 0},
    {"dataset": "hypersim", "scene_id": "ai_035_001", "save_id": 1},
    {"dataset": "custom", "scene_id": "exp5", "save_id": 1},
    {"dataset": "custom", "scene_id": "exp5", "save_id": 2},
    {"dataset": "custom", "scene_id": "exp5", "save_id": 3},
    {"dataset": "custom", "scene_id": "exp5", "save_id": 6},
]

################################### FID, KID ##########################################
metas = [
    {"dataset": "hypersim", "scene_id": "ai_001_001"},
    # {"dataset": "hypersim", "scene_id": "ai_001_008"},
    # {"dataset": "hypersim", "scene_id": "ai_010_004"},
    # {"dataset": "hypersim", "scene_id": "ai_010_004"},
    # {"dataset": "hypersim", "scene_id": "ai_010_004"},
    # {"dataset": "hypersim", "scene_id": "ai_010_005"},
    # {"dataset": "hypersim", "scene_id": "ai_012_005"},
    # {"dataset": "hypersim", "scene_id": "ai_035_001"},
]


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metric", type=str, default=None, required=True)
    parser.add_argument("--method", type=str, default="scenecraft", required=False)
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument("--num-workers", type=int, help="Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`")
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument("--dims", type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help="Dimensionality of Inception features to use. By default, uses pool3 features")
    parser.add_argument("--save-stats", action="store_true", help=(
            "Generate an npz archive from a directory of samples. The first path is used as input and the second as output."))
    parser.add_argument("--root", type=str, default=None, required=False)
    main()
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint

from clip import clip
from .coop import load_clip_to_cpu, TextEncoder, PromptLearner


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}.",
    "DescribableTextures": "a photo of a {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
}


class MyCustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # print(logit_scale)
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class TTPT(TrainerX):
    def build_model(self):
        cfg = self.cfg
        if cfg.DATASET.SUBSAMPLE_CLASSES == "base":
            classnames = self.dm.dataset.classnames_base + self.dm.dataset.classnames_new
        elif cfg.DATASET.SUBSAMPLE_CLASSES == "new":
            classnames = self.dm.dataset.classnames_new + self.dm.dataset.classnames_base

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model_base = MyCustomCLIP(cfg, self.dm.dataset.classnames_base, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model_base.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model_base.to(self.device)

        # print("Building custom CLIP")
        # self.model_new = MyCustomCLIP(cfg, self.dm.dataset.classnames_new, clip_model)

        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.model_new.named_parameters():
            # if "prompt_learner" not in name:
                # param.requires_grad_(False)

        # self.model_new.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tokenized_clip_prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_clip_prompts = torch.cat([clip.tokenize(p) for p in tokenized_clip_prompts])
        tokenized_clip_prompts = tokenized_clip_prompts.to(self.device)
        self.tokenized_clip_prompts = tokenized_clip_prompts

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_learner.to(self.device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = clip_model
        self.clip_model.to(self.device)

    def model_inference(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.model_base.logit_scale.exp()

        image_features = self.model_base.image_encoder(image.type(self.model_base.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        coop_prompts = self.prompt_learner()
        clip_prompts = self.clip_model.token_embedding(self.tokenized_clip_prompts).type(self.model_base.dtype)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        logits_base = self.model_base(image)
        mcm_scores_base = F.softmax(0.01 * logits_base, dim=1).max(dim=1)[0]
        # logits_new = self.model_new(image)
        # mcm_scores_new = F.softmax(logits_new, dim=1).max(dim=1)[0]
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        tokenized_clip_prompts_new = [temp.format(c.replace("_", " ")) for c in self.dm.dataset.classnames_new]
        tokenized_clip_prompts_new = torch.cat([clip.tokenize(p) for p in tokenized_clip_prompts_new])
        tokenized_clip_prompts_new = tokenized_clip_prompts_new.to(self.device)
        logits_per_image, _ = self.clip_model(image, tokenized_clip_prompts_new)
        mcm_scores_new = F.softmax(0.01 * logits_per_image, dim=1).max(dim=1)[0]
        for mcm_base_i, mcm_new_i in zip(mcm_scores_base, mcm_scores_new):
            # print("mcm_base_i", end=": ")
            # print(mcm_base_i)
            # print("mcm_new_i", end=": ")
            # print(mcm_new_i)
            pts_i = mcm_base_i / (mcm_base_i + mcm_new_i) * coop_prompts + mcm_new_i / (mcm_base_i + mcm_new_i) * clip_prompts  # (n_cls, n_tkn, ctx_dim)
            # pts_i = 0.5 * coop_prompts + 0.5 * clip_prompts  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.model_base.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # if self.cfg.DATASET.SUBSAMPLE_CLASSES == "base":
            # return torch.cat((logits_base, logits_per_image), dim=1)
        # elif self.cfg.DATASET.SUBSAMPLE_CLASSES == "new":
            # return torch.cat((logits_per_image, logits_base), dim=1)
        return logits

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = ["prompt_learner"]

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self.model_base.prompt_learner.load_state_dict(state_dict, strict=False)
            # self.model_new.prompt_learner.load_state_dict(state_dict, strict=False)
            self.prompt_learner.load_state_dict(state_dict, strict=False)

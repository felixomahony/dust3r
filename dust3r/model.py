# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float("inf")

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), (
    "Outdated huggingface_hub version, " "please reinstall requirements.txt"
)


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo(
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",  # PatchEmbedDust3R or ManyAR_PatchEmbed
        **croco_kwargs,
    ):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(
            output_mode,
            head_type,
            landscape_only,
            depth_mode,
            conf_mode,
            **croco_kwargs,
        )
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim
        )

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token],
            "encoder": [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        self.downstream_head2 = head_factory(
            head_type, output_mode, self, has_conf=bool(conf_mode)
        )
        # magic wrapper
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=landscape_only
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, *args):  # img1, img2, true_shape1, true_shape2):
        # *args are made up of two sets of objects of identical length: images and shapes
        objs_len = len(args) // 2
        imgs = tuple(args[:objs_len])
        true_shapes = tuple(args[objs_len:])
        # if img1.shape[-2:] == img2.shape[-2:]:
        if all(imgs[i].shape[-2:] == imgs[0].shape[-2:] for i in range(1, objs_len)):
            out, pos, _ = self._encode_image(
                # torch.cat((img1, img2), dim=0),
                torch.cat(imgs, dim=0),
                # torch.cat((true_shape1, true_shape2), dim=0),
                torch.cat(true_shapes, dim=0),
            )
            outs = out.chunk(objs_len, dim=0)
            poss = pos.chunk(objs_len, dim=0)
        else:
            encoder_outputs = tuple(
                self._encode_image(imgs[i], true_shapes[i]) for i in range(objs_len)
            )
            outs = tuple(encoder_outputs[i][0] for i in range(objs_len))
            poss = tuple(encoder_outputs[i][1] for i in range(objs_len))
            # out, pos, _ = self._encode_image(img1, true_shape1)
            # out2, pos2, _ = self._encode_image(img2, true_shape2)
        # return out, out2, pos, pos2
        return *outs, *poss

    def _encode_symmetrized(self, *views):
        # img1 = view1["img"]
        # img2 = view2["img"]
        imgs = tuple(view["img"] for view in views)
        B = imgs[0].shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        # shape1 = view1.get(
        #     "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        # )
        # shape2 = view2.get(
        #     "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        # )
        shapes = tuple(
            view.get("true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1))
            for view, img in zip(views, imgs)
        )
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(views[0], views[1]):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                imgs[0][::2], imgs[1][::2], shapes[0][::2], shapes[1][::2]
            )
            feats = tuple(interleave(feat1, feat2))
            poss = tuple(interleave(pos1, pos2))
        else:
            # feat1, feat2, pos1, pos2 = self._encode_image_pairs(
            #     imgs[0], imgs[1], shapes[0], shapes[1]
            # )
            feats_poss = tuple(self._encode_image_pairs(*imgs, *shapes))
            feats = feats_poss[: len(feats_poss) // 2]
            poss = feats_poss[len(feats_poss) // 2 :]

        return shapes, feats, poss

    def _decoder(self, *f1_and_pos, return_attention=False):  # f1, pos1, f2, pos2):
        # f1_and_pos = (f1, pos1, f2, pos2, ..., fn, posn)
        n = len(f1_and_pos) // 2
        # final_output = [(f1, f2)]  # before projection
        final_output = [tuple(f1_and_pos[2 * i] for i in range(n))]

        # project to decoder dim
        # f1 = self.decoder_embed(f1)
        # f2 = self.decoder_embed(f2)
        fs = tuple(self.decoder_embed(f1_and_pos[2 * i]) for i in range(n))

        poss = tuple(f1_and_pos[2 * i + 1] for i in range(n))

        # final_output.append((f1, f2))
        final_output.append(fs)

        attn_flag = return_attention
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            f1, _ = blk1(
                final_output[-1][0],
                final_output[-1][1:],
                poss[0],
                poss[1:],
                return_attention=attn_flag,
            )
            if attn_flag:
                f1, attn = f1
            fns = tuple(
                blk2(final_output[-1][i], final_output[-1][0], poss[i], poss[0])[0]
                for i in range(1, n)
            )
            # store the result
            final_output.append((f1, *fns))

            attn_flag = False  # only return attention for the first block

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        if return_attention:
            return zip(*final_output), attn
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(self, *views):
        # encode the two images --> B,S,D
        shapes, feats, poss = self._encode_symmetrized(*views)
        # (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        decs = tuple(
            self._decoder(*tuple(item for pair in zip(feats, poss) for item in pair))
        )
        # dec1, dec2 = decs
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.amp.autocast("cuda", enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in decs[0]], shapes[0])
            res2s = tuple(
                self._downstream_head(2, [tok.float() for tok in decs[i]], shapes[i])
                for i in range(1, len(decs))
            )

        for i in range(len(res2s)):
            res2s[i]["pts3d_in_other_view"] = res2s[i].pop(
                "pts3d"
            )  # predict view2's pts3d in view1's frame
        return res1, *res2s

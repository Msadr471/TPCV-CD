# In change_classifier.py
from typing import List
import torchvision
from models.layers import MixingMaskAttentionBlock, PixelwiseLinear, UpMask, MixingBlock, RetinaSimBlock
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid
from torchvision.models import EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights


class ChangeClassifier(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        weights=None,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()

        self._retina = RetinaSimBlock(in_channels=3, out_channels=3, kernel_size=15)

        # Set default weights if not specified
        if weights is None:
            if bkbn_name == "efficientnet_b4":
                weights = EfficientNet_B4_Weights.DEFAULT
            elif bkbn_name == "efficientnet_b5":
                weights = EfficientNet_B5_Weights.DEFAULT
            elif bkbn_name == "efficientnet_b6":
                weights = EfficientNet_B6_Weights.DEFAULT
            elif bkbn_name == "efficientnet_b7":
                weights = EfficientNet_B7_Weights.DEFAULT
            else:
                weights = EfficientNet_B4_Weights.DEFAULT

        # Load the weights backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, weights, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks based on backbone type:
        if bkbn_name == "efficientnet_b4":
            mixing_blocks = [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
            up_dims = [(2, 56, 64), (2, 64, 64), (2, 64, 32)]
            
        elif bkbn_name == "efficientnet_b5":
            mixing_blocks = [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(80, 40, [40, 20, 10], [20, 10, 1]),
                MixingBlock(128, 64),
            ]
            up_dims = [(2, 64, 80), (2, 80, 80), (2, 80, 32)]
            
        elif bkbn_name == "efficientnet_b6":
            mixing_blocks = [
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingMaskAttentionBlock(80, 40, [40, 20, 10], [20, 10, 1]),
                MixingBlock(144, 72),
            ]
            up_dims = [(2, 72, 80), (2, 80, 80), (2, 80, 32)]
            
        elif bkbn_name == "efficientnet_b7":
            mixing_blocks = [
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingMaskAttentionBlock(96, 48, [48, 24, 12], [24, 12, 1]),
                MixingBlock(160, 80),
            ]
            up_dims = [(2, 80, 96), (2, 96, 96), (2, 96, 32)]
            
        else:
            # Default to B4
            mixing_blocks = [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
            up_dims = [(2, 56, 64), (2, 64, 64), (2, 64, 32)]

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(mixing_blocks)

        # Initialize Upsampling blocks:
        self._up = ModuleList([UpMask(*dims) for dims in up_dims])

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = self._encode(ref, test)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, ref, test) -> List[Tensor]:
        ref = self._retina(ref)
        test = self._retina(test)
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:  # Skip layer 0 (the stem)
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


def _get_backbone(
    bkbn_name, weights, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        weights=weights
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model
from typing import List
import torch
import torchvision
from models.layers import MixingMaskAttentionBlock, PixelwiseLinear, UpMask, MixingBlock, RetinaSimBlock
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid

class ChangeClassifier(Module):
    def __init__(
        self,
        weights=None,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()

        self._retina = RetinaSimBlock(in_channels=3, out_channels=3, kernel_size=15)

        # Set default weights to EfficientNet_B4_Weights.DEFAULT
        if weights is None:
            from torchvision.models import EfficientNet_B4_Weights
            weights = EfficientNet_B4_Weights.DEFAULT

        # Load the backbone (hardcoded to efficientnet_b4)
        self._backbone = _get_backbone(
            weights, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks (hardcoded for efficientnet_b4)
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList([
            MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
            MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
            MixingBlock(112, 56),
        ])
        up_dims = [(2, 56, 64), (2, 64, 64), (2, 64, 32)]

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
    weights, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # Hardcoded to efficientnet_b4
    with torch.no_grad():  # Prevent tracking gradients for the backbone
        model = torchvision.models.efficientnet_b4(weights=weights)
        features = model.features
        
        # Slicing it:
        derived_model = ModuleList([])
        for name, layer in features.named_children():
            derived_model.append(layer)
            if name == output_layer_bkbn:
                break

        # Freezing the backbone weights:
        if freeze_backbone:
            for param in derived_model.parameters():
                param.requires_grad = False
        
        # Clear memory by deleting the full model and forcing garbage collection
        del model
        del features
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return derived_model
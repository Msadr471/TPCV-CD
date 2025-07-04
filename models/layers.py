from typing import List, Optional

from torch import Tensor, reshape, stack, meshgrid, exp, arange

from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
    Tanh
)


class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)




class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class RetinaSimBlock(Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=15):
        super().__init__()
        self.dog = Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)
        self.adapt = InstanceNorm2d(out_channels)
        self.act = Tanh()

        sigma_center = 1.0
        sigma_surround = 2.0
        self._init_dog_weights(sigma_center, sigma_surround)
        assert in_channels == out_channels, "RetinaSimBlock requires in_channels == out_channels for depthwise conv."

    def _init_dog_weights(self, sigma_c, sigma_s):
        def gaussian(sigma, k):
            ax = arange(-k // 2 + 1., k // 2 + 1.)
            xx, yy = meshgrid(ax, ax, indexing='ij')
            kernel = exp(-(xx**2 + yy**2) / (2. * sigma**2))
            return kernel / kernel.sum()

        ksize = self.dog.kernel_size[0]
        center = gaussian(sigma_c, ksize)
        surround = gaussian(sigma_s, ksize)
        dog_filter = center - surround
        for i in range(self.dog.out_channels):
            self.dog.weight.data[i, 0, :, :] = dog_filter

        self.dog.weight.requires_grad = False

    def forward(self, x):
        x = self.dog(x)
        x = self.adapt(x)
        return self.act(x)
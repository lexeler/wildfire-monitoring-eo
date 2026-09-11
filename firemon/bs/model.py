"""Сети сегментации области пожара (segmentation_models_pytorch): U-Net, FPN и др.
на 83 каналах (94 с каналами дальнего контекста)."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch

from .features import n_features


class CoordInput(torch.nn.Module):
    """Добавляет ко входу два канала положения в чипе: расстояние от центра и
    «квадратное» расстояние до края (оба инвариантны к поворотам/отражениям,
    поэтому совместимы с диэдральной аугментацией и TTA). Чипы вырезаны вокруг
    события: доля гари среди допустимых пикселей падает от центра к углам."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([torch.sqrt(yy * yy + xx * xx), torch.maximum(yy.abs(), xx.abs())])
        return self.net(torch.cat([x, pos.expand(n, -1, -1, -1)], 1))


def build_model(arch: str = "unet", encoder: str = "resnet34", pretrained: bool = False, classes: int = 4, ctx: int = 0,
                coord: bool = False, dec_attn: str | None = None, drop_path: float = 0.0,
                aux_head: bool = False, hot: bool = False) -> torch.nn.Module:
    """pretrained=True — веса ImageNet для энкодера (только при обучении;
    при инференсе веса целиком берутся из нашего чекпойнта).
    hot — каналы «видимый огонь» (features.hot_features); coord — каналы положения (CoordInput); dec_attn="scse" — внимание в
    декодере U-Net/UNet++; drop_path — stochastic depth энкодеров timm (tu-*)."""
    weights = "imagenet" if pretrained else None
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=n_features(ctx, hot) + (2 if coord else 0), classes=classes)
    if dec_attn:
        kw["decoder_attention_type"] = dec_attn
    if drop_path > 0:
        kw["drop_path_rate"] = drop_path
    if aux_head:  # вспомогательная голова уровня чипа (только обучение, см. train.chip_fraction_loss)
        kw["aux_params"] = dict(classes=1, pooling="avg", dropout=0.2)
    archs = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN, "manet": smp.MAnet,
             "deeplabv3p": smp.DeepLabV3Plus, "linknet": smp.Linknet}
    if arch not in archs:
        raise ValueError(arch)
    net = archs[arch](**kw)
    return CoordInput(net) if coord else net


def init_stem_from_bands(model: torch.nn.Module, mode: str) -> None:
    """Initialise the pretrained first conv from physical spectral channels.

    timm/SMP adapt a 3-channel ImageNet stem to N input channels by repeating the
    RGB kernels over all channels and scaling by 3/N. Here the three pretrained
    kernels are instead placed on three chosen Sentinel-2 bands of the "pre" and
    "post" images (half weight each) and every other input channel starts from
    zero: rgb_pair -> B4, B3, B2; swir_pair -> B12, B11, B8A (the bands of NBR).
    Channel layout: 0..8 "pre" B2..B12, 9..17 "post" (features.compute_features).
    """
    net = model.net if isinstance(model, CoordInput) else model
    stem = next(m for m in net.encoder.modules() if isinstance(m, torch.nn.Conv2d))
    if stem.in_channels < 27 or stem.groups != 1:
        raise ValueError(f"Unexpected input stem: {stem}")
    bands = {"rgb_pair": (2, 1, 0), "swir_pair": (8, 7, 6)}[mode]
    with torch.no_grad():
        rgb = stem.weight[:, :3].clone() * (stem.in_channels / 3.0)  # undo the 3/N scaling of adapt_input_conv
        stem.weight.zero_()
        for colour, band in enumerate(bands):
            stem.weight[:, band] = rgb[:, colour] * 0.5
            stem.weight[:, 9 + band] = rgb[:, colour] * 0.5

"""PanoCompactGS hash-grid conditional entropy model adapted to Phase6.4 attributes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import ATTRIBUTE_DIM
from .groups import GROUP_LAYOUT, GROUP_ORDER
from .backend import Entropy_gaussian_mix_prob_2, GridEncoder, STE_binary


@dataclass(frozen=True)
class ModelConfig:
    resolutions: tuple[int, ...] = (8, 12, 18, 27, 40, 60)
    resolutions_2d: tuple[int, ...] = (16, 32, 64, 128)
    features_per_level: int = 2
    log2_hashmap_size: int = 12
    log2_hashmap_size_2d: int = 12
    spherical_resolutions: tuple[int, ...] = (8, 16, 32, 64)
    log2_hashmap_size_spherical: int = 11
    spherical_feature_dim: int = 4
    hidden_dim: int = 64
    prefix_dim: int = 24
    symbol_clip: float = 15000.0
    group_order: tuple[str, ...] = GROUP_ORDER
    attribute_dim: int = ATTRIBUTE_DIM

    def to_dict(self) -> dict:
        result = asdict(self)
        result["resolutions"] = list(self.resolutions)
        result["resolutions_2d"] = list(self.resolutions_2d)
        result["spherical_resolutions"] = list(self.spherical_resolutions)
        result["group_order"] = list(self.group_order)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        value = dict(value)
        value["resolutions"] = tuple(int(x) for x in value["resolutions"])
        value["resolutions_2d"] = tuple(int(x) for x in value["resolutions_2d"])
        value["spherical_resolutions"] = tuple(int(x) for x in value["spherical_resolutions"])
        value["group_order"] = tuple(str(x) for x in value["group_order"])
        return cls(**value)


class EntropyModel(nn.Module):
    """PanoCompactGS spatial hash plus causal intra-Gaussian semantic group context."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # This is PanoCompactGS's structured hash grid: one 3D grid plus the XY/XZ/YZ
        # planar grids from upstream mix_3D2D_encoding.
        self.grid_xyz = GridEncoder(
            num_dim=3,
            n_features=config.features_per_level,
            resolutions_list=config.resolutions,
            log2_hashmap_size=config.log2_hashmap_size,
            ste_binary=True,
            ste_multistep=False,
            add_noise=False,
            Q=1,
        )
        plane_kwargs = dict(
            num_dim=2,
            n_features=config.features_per_level,
            resolutions_list=config.resolutions_2d,
            log2_hashmap_size=config.log2_hashmap_size_2d,
            ste_binary=True,
            ste_multistep=False,
            add_noise=False,
            Q=1,
        )
        self.grid_xy = GridEncoder(**plane_kwargs)
        self.grid_xz = GridEncoder(**plane_kwargs)
        self.grid_yz = GridEncoder(**plane_kwargs)
        self.grid_spherical = GridEncoder(
            num_dim=3,
            n_features=config.features_per_level,
            resolutions_list=config.spherical_resolutions,
            log2_hashmap_size=config.log2_hashmap_size_spherical,
            ste_binary=True,
            ste_multistep=False,
            add_noise=False,
            Q=1,
        )
        context_dim = (
            self.grid_xyz.output_dim
            + self.grid_xy.output_dim
            + self.grid_xz.output_dim
            + self.grid_yz.output_dim
            + self.grid_spherical.output_dim
            + 1
        )
        if config.attribute_dim != ATTRIBUTE_DIM:
            raise ValueError(f"attribute_dim must remain {ATTRIBUTE_DIM}")
        if config.spherical_feature_dim != 4:
            raise ValueError("spherical_feature_dim must remain 4 (direction xyz + log radius)")
        if tuple(config.group_order) != GROUP_ORDER:
            raise ValueError(f"group_order must remain {GROUP_ORDER}")
        if config.prefix_dim <= 0 or config.hidden_dim <= 0 or config.symbol_clip <= 0:
            raise ValueError("hidden_dim, prefix_dim and symbol_clip must be positive")
        self.context_trunk = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.prefix_projections = nn.ModuleList(
            nn.Linear(spec.width, config.prefix_dim, bias=False) for spec in GROUP_LAYOUT
        )
        # A single linear head per semantic group keeps the decoder smaller
        # than six independent two-layer MLPs while still allowing each later
        # group to use an exact decoded prefix.
        head_input_dim = config.hidden_dim + config.prefix_dim + 2
        self.group_heads = nn.ModuleList(
            nn.Linear(head_input_dim, spec.width * 5) for spec in GROUP_LAYOUT
        )
        self.entropy = Entropy_gaussian_mix_prob_2(Q=1)

    def spatial_context(
        self,
        xyz_normalized: torch.Tensor,
        spherical_features: torch.Tensor,
    ) -> torch.Tensor:
        if xyz_normalized.ndim != 2 or xyz_normalized.shape[1] != 3:
            raise ValueError("xyz_normalized must have shape (N, 3)")
        if spherical_features.ndim != 2 or spherical_features.shape != (
            xyz_normalized.shape[0],
            self.config.spherical_feature_dim,
        ):
            raise ValueError("spherical_features must have shape (N, 4)")
        if not torch.isfinite(xyz_normalized).all() or not torch.isfinite(
            spherical_features
        ).all():
            raise ValueError("spatial context inputs must be finite")
        x, y, z = xyz_normalized.split(1, dim=-1)
        direction_normalized = (spherical_features[:, :3] + 1.0) * 0.5
        log_radius = spherical_features[:, 3:4]
        raw_context = torch.cat(
            (
                self.grid_xyz(xyz_normalized),
                self.grid_xy(torch.cat((x, y), dim=-1)),
                self.grid_xz(torch.cat((x, z), dim=-1)),
                self.grid_yz(torch.cat((y, z), dim=-1)),
                self.grid_spherical(direction_normalized),
                log_radius,
            ),
            dim=-1,
        )
        return self.context_trunk(raw_context)

    def initial_prefix(self, spatial_context: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (spatial_context.shape[0], self.config.prefix_dim),
            dtype=spatial_context.dtype,
            device=spatial_context.device,
        )

    def group_distribution(
        self,
        spatial_context: torch.Tensor,
        prefix: torch.Tensor,
        sh_levels: torch.Tensor,
        hard_flags: torch.Tensor,
        group_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= group_index < len(GROUP_LAYOUT):
            raise IndexError(f"group_index {group_index} is outside the semantic group layout")
        levels = sh_levels.reshape(-1, 1).to(spatial_context.dtype) / 3.0
        hard = hard_flags.reshape(-1, 1).to(spatial_context.dtype)
        inputs = torch.cat((spatial_context, prefix, levels, hard), dim=-1)
        output = self.group_heads[group_index](inputs)
        mean1, raw_scale1, mean2, raw_scale2, mixture_logit = output.chunk(5, dim=-1)
        mixture = torch.softmax(
            torch.stack((mixture_logit, torch.zeros_like(mixture_logit)), dim=-1), dim=-1
        )
        return (
            mean1,
            F.softplus(raw_scale1) + 1e-6,
            mean2,
            F.softplus(raw_scale2) + 1e-6,
            mixture[..., 0],
            mixture[..., 1],
        )

    def update_prefix(
        self,
        prefix: torch.Tensor,
        quantized_group: torch.Tensor,
        q_group: torch.Tensor,
        active_group: torch.Tensor,
        group_index: int,
    ) -> torch.Tensor:
        """Consume only integer symbols that the decoder already recovered."""

        symbols = torch.round(quantized_group.detach() / q_group.detach())
        symbols = torch.where(active_group, symbols, torch.zeros_like(symbols))
        # STE_multistep uses a fixed +/-15000-symbol bound.  Signed-log
        # normalization preserves distinctions across that entire legal
        # interval without letting extreme symbols dominate the prefix.
        normalized = torch.sign(symbols) * torch.log1p(symbols.abs())
        normalized = torch.clamp(
            normalized / math.log1p(self.config.symbol_clip), -1.0, 1.0
        )
        update = self.prefix_projections[group_index](normalized)
        return torch.tanh(prefix + update)

    def estimated_bits(
        self,
        quantized: torch.Tensor,
        xyz_normalized: torch.Tensor,
        spherical_normalized: torch.Tensor,
        q: torch.Tensor,
        sh_levels: torch.Tensor,
        hard_flags: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.spatial_context(xyz_normalized, spherical_normalized)
        prefix = self.initial_prefix(context)
        bits = torch.zeros_like(quantized)
        for group_index, spec in enumerate(GROUP_LAYOUT):
            selection = spec.source_slice
            parameters = self.group_distribution(
                context, prefix, sh_levels, hard_flags, group_index
            )
            mean1, scale1, mean2, scale2, prob1, prob2 = parameters
            group_bits = self._group_entropy_bits(
                quantized[:, selection],
                mean1,
                scale1,
                mean2,
                scale2,
                prob1,
                prob2,
                q[:, selection],
            )
            bits[:, selection] = group_bits
            prefix = self.update_prefix(
                prefix,
                quantized[:, selection],
                q[:, selection],
                active_mask[:, selection],
                group_index,
            )
        return bits

    def _group_entropy_bits(
        self,
        values: torch.Tensor,
        mean1: torch.Tensor,
        scale1: torch.Tensor,
        mean2: torch.Tensor,
        scale2: torch.Tensor,
        prob1: torch.Tensor,
        prob2: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate likelihoods with the same serialized zero symbol origin as coding."""

        return self.entropy(
            values,
            mean1,
            mean2,
            scale1,
            scale2,
            prob1,
            prob2,
            Q=q,
            x_mean=torch.zeros_like(q),
        )

    def forward(
        self,
        quantized: torch.Tensor,
        xyz_normalized: torch.Tensor,
        spherical_normalized: torch.Tensor,
        q: torch.Tensor,
        sh_levels: torch.Tensor,
        hard_flags: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.estimated_bits(
            quantized,
            xyz_normalized,
            spherical_normalized,
            q,
            sh_levels,
            hard_flags,
            active_mask,
        )

    def binary_hash(self) -> torch.Tensor:
        parameters = tuple(grid.params for grid in self.hash_grids())
        if not torch.stack([torch.isfinite(value).all() for value in parameters]).all().item():
            raise ValueError("hash-grid parameters must be finite before binarization")
        result = torch.cat(
            [STE_binary.apply(value).reshape(-1) for value in parameters]
        ).detach()
        if not torch.all((result == -1) | (result == 1)).item():
            raise RuntimeError("binarized hash must contain only -1 and +1")
        return result

    def hash_grids(self) -> tuple[GridEncoder, ...]:
        return self.grid_xyz, self.grid_xy, self.grid_xz, self.grid_yz, self.grid_spherical

    def deployment_state(self, *, fp16: bool = False) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            # Hash parameters have their own binary stream.  Grid offsets and
            # resolutions are structural buffers derived from the canonical
            # ModelConfig and must never be overwritten by untrusted payloads.
            if key.endswith(
                (".params", ".offsets_list", ".resolutions_list")
            ):
                continue
            value = value.detach().cpu()
            if fp16 and value.is_floating_point():
                value = value.to(torch.float16)
            result[key] = value
        return result

    @staticmethod
    def _deployment_value(parameter: torch.Tensor, mode: str) -> torch.Tensor:
        source = parameter.detach()
        if mode == "int8" and source.ndim >= 2:
            rows = source.reshape(source.shape[0], -1)
            maximum = rows.abs().amax(dim=1, keepdim=True)
            scale = torch.where(
                maximum == 0,
                torch.ones_like(maximum),
                maximum / 127.0,
            )
            return (torch.round(rows / scale).clamp(-127, 127) * scale).reshape_as(source)
        return source.to(torch.float16).to(source.dtype)

    def fake_quantized_parameter_overrides(self, mode: str) -> dict[str, torch.Tensor]:
        """Return deployed-value forwards with identity gradients to FP32 masters."""

        if mode not in {"int8", "fp16"}:
            raise ValueError("deployment mode must be 'int8' or 'fp16'")
        overrides: dict[str, torch.Tensor] = {}
        for name, parameter in self.named_parameters():
            if name.endswith(".params"):
                continue
            deployed = self._deployment_value(parameter, mode)
            overrides[name] = parameter + (deployed - parameter).detach()
        return overrides

    @torch.no_grad()
    def project_deployment_weights(self, mode: str) -> None:
        """Destructively project weights for diagnostics, never for training.

        The real Phase8.8 training path keeps FP32 masters and calls
        :meth:`fake_quantized_parameter_overrides`; this helper remains useful
        only for explicit projection tests. Binary hash-table parameters keep
        PanoCompactGS's dedicated STE and bitstream.
        """

        if mode not in {"int8", "fp16"}:
            raise ValueError("deployment mode must be 'int8' or 'fp16'")
        named_parameters = tuple(self.named_parameters())
        if not torch.stack(
            [torch.isfinite(parameter).all() for _, parameter in named_parameters]
        ).all().item():
            raise ValueError("all model parameters must be finite before deployment projection")
        for name, parameter in named_parameters:
            if name.endswith(".params"):
                continue
            if not parameter.is_floating_point():
                continue
            parameter.copy_(self._deployment_value(parameter, mode))
        if not torch.stack(
            [torch.isfinite(parameter).all() for _, parameter in named_parameters]
        ).all().item():
            raise ValueError("deployment projection produced a non-finite model parameter")

    @torch.no_grad()
    def install_binary_hash(self, values: torch.Tensor) -> None:
        values = values.reshape(-1)
        expected = sum(grid.params.numel() for grid in self.hash_grids())
        if values.numel() != expected:
            raise ValueError(f"Hash stream has {values.numel()} entries; expected {expected}")
        if not torch.isfinite(values).all().item() or not torch.all(
            (values == -1) | (values == 1)
        ).item():
            raise ValueError("Hash stream must contain only finite -1/+1 values")
        offset = 0
        for grid in self.hash_grids():
            count = grid.params.numel()
            grid.params.copy_(values[offset : offset + count].view_as(grid.params).to(grid.params))
            offset += count
        if offset != values.numel():  # defensive invariant
            raise RuntimeError(f"Hash stream has {values.numel()} entries; model consumed {offset}")

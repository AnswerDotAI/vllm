from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class TorchaoConfig(QuantizationConfig):
    """Config class for torchao _weight_int4pack_mm.

    Reference: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu
    """

    def __init__(
        self,
        group_size: int,
        inner_k_tiles: int,
        lora_rank: int = None,
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size
        self.inner_k_tiles = inner_k_tiles
        self.lora_rank = lora_rank
        if self.group_size not in [32, 64, 128, 256]:
            raise ValueError(
                "Currently, only group sizes [32, 64, 128, 256] "
                "are supported for _weight_int4pack_mm_cuda, but got group_size of "
                f"{self.group_size}")

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // 4

    def __repr__(self) -> str:
        return f"TorchaoConfig(group_size={self.group_size})"

    @classmethod
    def get_name(cls) -> str:
        return "torchao"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TorchaoConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        inner_k_tiles = config.get("inner_k_tiles", 8)
        lora_rank = config.get("lora_rank", None)
        return cls(group_size, inner_k_tiles, lora_rank)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["TorchaoLinearMethod"]:
        if isinstance(layer, LinearBase):
            if self.lora_rank is None:
                return TorchaoLinearMethod(self)
            else:
                return TorchaoDORALinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class TorchaoLinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        quant_config: The torchao quantization config.
    """

    def __init__(self, quant_config: TorchaoConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.

        if params_dtype != torch.bfloat16:
            raise ValueError(
                f"The params dtype must be bfloat16, but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}.")

        if (input_size_per_partition % self.quant_config.group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        # Quantized 4Bit weights packed into Int32.
        qweight = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.pack_factor,
                input_size_per_partition // (self.quant_config.inner_k_tiles * 16),
                32,
                self.quant_config.inner_k_tiles // 2,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales_and_zeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                2,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales_and_zeros,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales_and_zeros", scales_and_zeros)
        set_weight_attrs(scales_and_zeros, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales_and_zeros = layer.scales_and_zeros
        output_size = qweight.size(0) * self.quant_config.pack_factor
        
        origin_x_size = x.size()
        x_reshaped = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x_reshaped, qweight, self.quant_config.group_size, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (output_size,)
        output = c.reshape(new_shape)        

        if bias is not None:
            output.add_(bias)  # In-place add

        return output


class TorchaoDORALinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        quant_config: The torchao quantization config.
    """

    def __init__(self, quant_config: TorchaoConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.

        if params_dtype != torch.bfloat16:
            raise ValueError(
                f"The params dtype must be bfloat16, but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}.")

        if (input_size_per_partition % self.quant_config.group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        # Quantized 4Bit weights packed into Int32.
        qweight = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.pack_factor,
                input_size_per_partition // (self.quant_config.inner_k_tiles * 16),
                32,
                self.quant_config.inner_k_tiles // 2,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales_and_zeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                2,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales_and_zeros,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales_and_zeros", scales_and_zeros)
        set_weight_attrs(scales_and_zeros, extra_weight_attrs)
        
        # DORA params.
        lora_A = Parameter(
            torch.empty(
                self.quant_config.lora_rank,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("lora_A", lora_A)
        set_weight_attrs(
            lora_A, {
                "input_dim": 1,
            })
        set_weight_attrs(lora_A, extra_weight_attrs)
        
        lora_B = Parameter(
            torch.empty(
                output_size_per_partition,
                self.quant_config.lora_rank,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("lora_B", lora_B)
        set_weight_attrs(
            lora_B, {
                "output_dim": 0,
            })
        set_weight_attrs(lora_B, extra_weight_attrs)
        
        # precomputed m / || W + AB ||, where m is the magnitude parameter.
        rescale = Parameter(
            torch.empty(
                output_size_per_partition,
            ),
            requires_grad=False,
        )
        layer.register_parameter("rescale", rescale)
        set_weight_attrs(
            rescale, {
                "output_dim": 0,
            })        
        set_weight_attrs(rescale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales_and_zeros = layer.scales_and_zeros
        output_size = qweight.size(0) * self.quant_config.pack_factor
        lora_A, lora_B, rescale = layer.lora_A, layer.lora_B, layer.rescale
        
        
        origin_x_size = x.size()
        x_reshaped = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x_reshaped, qweight, self.quant_config.group_size, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (output_size,)
        output = c.reshape(new_shape)        
        
        # rescale 
        output = rescale.view(1,-1) * (output + (x @ lora_A.t() @ lora_B.t())) 
        

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

from typing import Union, Dict

logger = init_logger(__name__)

import os
from gemlite.core import DType, GemLiteLinear 

class GemLiteConfig(QuantizationConfig):
    """Config class for ...
    """

    def __init__(
        self,
        group_size: Union[int, Dict],
        nbits: Union[int, Dict],
        lora_rank: int = None,
        skipped_dora_layers: List[str] = [],
        block_influence_layers: List[str] = [],
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size
        self.lora_rank = lora_rank
        self.skipped_dora_layers = skipped_dora_layers
        self.block_influence_layers = block_influence_layers
        
        # if isinstance(group_size, int):        
        #     if self.group_size not in [32, 64, 128, 256]:
        #         raise ValueError(
        #             "Currently, only group sizes [32, 64, 128, 256] "
        #             "are supported for _weight_int4pack_mm_cuda, but got group_size of "
        #             f"{self.group_size}")
        # elif isinstance(group_size, Dict):
        #     # {layer_name: group_size}
        #     for k,v in group_size.items():
        #         if v not in [32, 64, 128, 256]:
        #             raise ValueError(
        #                 "Currently, only group sizes [32, 64, 128, 256] "
        #                 "are supported for _weight_int4pack_mm_cuda, but got group_size of "
        #                 f"{v}")
        
        # 4 Bits packed into 8 bit datatype
        self.nbits = nbits
        self.pack_bit = 32
        
        if isinstance(nbits, int):
            self.pack_factor = self.pack_bit // nbits
        elif isinstance(nbits, Dict):
            # {layer_name: nbits}
            self.pack_factor = {k: self.pack_bit // v for k,v in nbits.items()}
                
    def __repr__(self) -> str:
        return f"GemLiteConfig(group_size={self.group_size})"

    @classmethod
    def get_name(cls) -> str:
        return "gemlite"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GemLiteConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        nbits = cls.get_from_keys(config, ["nbits"])
        lora_rank = config.get("lora_rank", None)
        skipped_dora_layers = config.get("skipped_dora_layers", [])
        block_influence_layers = config.get("block_influence_layers", [])
        return cls(group_size, nbits, lora_rank, skipped_dora_layers, block_influence_layers)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix:str) -> Optional["GemLiteLinearMethod"]:
        if isinstance(layer, LinearBase):
            
            print(f"Getting Quant Method for {prefix}")
            print(f"Block Influence layers: {self.block_influence_layers}")
            
            if self.lora_rank is None or any(l in prefix for l in self.skipped_dora_layers):
                print(f"Using GemLiteLinearMethod for skipped: {prefix}")
                return GemLiteLinearMethod(self)
            elif any(l + "." in prefix for l in self.block_influence_layers):
                print(f"Using GemLiteDORALinearMethod for block influence: {prefix}")
                return GemLiteDORALinearMethod(self, is_block_influence=True)
            else:
                return GemLiteDORALinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class GemLiteLinearMethod(LinearMethodBase):
    """Linear method for gemlite. Supporting mixed 4/2 bits.

    Args:
        quant_config: The gemlite quantization config.
    """
    
    def __init__(self, quant_config: GemLiteConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_name: str,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        
        if isinstance(self.quant_config.nbits, int):
            self.layer_nbits = self.quant_config.nbits
            self.layer_pack_factor = self.quant_config.pack_factor
        elif isinstance(self.quant_config.nbits, Dict):
            # {layer_name: nbits}
            self.layer_nbits = self.quant_config.nbits[layer_name]            
            self.layer_pack_factor = self.quant_config.pack_factor[layer_name]
        
        if isinstance(self.quant_config.group_size, int):
            self.layer_group_size = self.quant_config.group_size
        elif isinstance(self.quant_config.group_size, Dict):
            self.layer_group_size = self.quant_config.group_size[layer_name] 
        
        if params_dtype != torch.half:
            raise ValueError(
                f"The params dtype must be half, but got {params_dtype}")
        
        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.layer_pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.layer_pack_factor}.")

        if (input_size_per_partition % self.layer_group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_pack_factor,
                output_size_per_partition,
                device="cuda",
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.layer_pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_group_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        zeros = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_group_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            zeros,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("zeros", zeros)
        set_weight_attrs(zeros, extra_weight_attrs)
        
        
        K = input_size_per_partition # this is the dequantized input size
        N = output_size_per_partition # this is the dequantized output size
        self.gemlite_linear = GemLiteLinear(
            self.layer_nbits, #supported: [8, 4, 2, 1]
            group_size=self.layer_group_size, # any group_size divisible by 32
            in_features=K, # input size
            out_features=N, #ouput size
            input_dtype=DType.FP16, #FP16 or BF16
            output_dtype=DType.FP16, #FP16 or BF16
            acc_dtype=DType.FP16, #FP16 or FP32 
        )

        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Init gemlite linear weights.
        if not hasattr(self.gemlite_linear, "W_q"):
            self.gemlite_linear.W_q = layer.qweight
            self.gemlite_linear.scales = layer.scales
            self.gemlite_linear.zeros = layer.zeros
        
        # x : bs x seq_len x hidden_size
        output = self.gemlite_linear(x)
        
        if bias is not None:
            output.add_(bias)  # In-place add

        return output


class GemLiteDORALinearMethod(LinearMethodBase):
    """Linear method for gemlite. Supporting mixed 4/2 bits.

    Args:
        quant_config: The gemlite quantization config.
    """
    
    def __init__(self, quant_config: GemLiteConfig, is_block_influence=False):
        self.quant_config = quant_config
        self.is_block_influence = is_block_influence

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_name: str,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        
        if self.is_block_influence:
            # block influence layers use higher bit precision.
            self.layer_nbits = 4
            self.layer_pack_factor = 2
        elif isinstance(self.quant_config.nbits, int):
            self.layer_nbits = self.quant_config.nbits
            self.layer_pack_factor = self.quant_config.pack_factor
        elif isinstance(self.quant_config.nbits, Dict):
            # {layer_name: nbits}
            self.layer_nbits = self.quant_config.nbits[layer_name]            
            self.layer_pack_factor = self.quant_config.pack_factor[layer_name]
        
        if self.is_block_influence:
            self.layer_group_size = 128 # FIXME: hardcoded for 4bit now
        elif isinstance(self.quant_config.group_size, int):
            self.layer_group_size = self.quant_config.group_size
        elif isinstance(self.quant_config.group_size, Dict):
            self.layer_group_size = self.quant_config.group_size[layer_name] 
        
        if params_dtype != torch.half:
            raise ValueError(
                f"The params dtype must be half, but got {params_dtype}")
        
        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.layer_pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.layer_pack_factor}.")

        if (input_size_per_partition % self.layer_group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_pack_factor,
                output_size_per_partition,
                device="cuda",
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.layer_pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_group_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        zeros = Parameter(
            torch.empty(
                input_size_per_partition // self.layer_group_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            zeros,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("zeros", zeros)
        set_weight_attrs(zeros, extra_weight_attrs)
        
        # TODO: BF16 support.
        K = input_size_per_partition # this is the dequantized input size
        N = output_size_per_partition # this is the dequantized output size
        self.gemlite_linear = GemLiteLinear(
            self.layer_nbits, #supported: [8, 4, 2, 1]
            group_size=self.layer_group_size, # any group_size divisible by 32
            in_features=K, # input size
            out_features=N, #ouput size
            input_dtype=DType.FP16, #FP16 or BF16
            output_dtype=DType.FP16, #FP16 or BF16
            acc_dtype=DType.FP16, #FP16 or FP32 
        )
        
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
                
    # TODO: Figure out why fullgraph=True exceeds the limit.
    # https://discuss.pytorch.org/t/torch-compile-cache-size-limit-best-practice/200713
    def dora_layer(self, x, output, rescale, lora_A, lora_B):
        return rescale.view(1,-1) * (output + x @ lora_A.t() @ lora_B.t()) 
        

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Init gemlite linear weights.
        if not hasattr(self.gemlite_linear, "W_q"):
            self.gemlite_linear.W_q = layer.qweight
            self.gemlite_linear.scales = layer.scales
            self.gemlite_linear.zeros = layer.zeros
            
        lora_A, lora_B, rescale = layer.lora_A, layer.lora_B, layer.rescale
        
        # x : bs x seq_len x hidden_size
        output = self.gemlite_linear(x)
                
        # rescale 
        # output = rescale.view(1,-1) * (output + x @ lora_A.t() @ lora_B.t()) 
        output = self.dora_layer(x, output, rescale, lora_A, lora_B)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

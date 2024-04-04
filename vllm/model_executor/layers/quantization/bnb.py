from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit, QuantState
import copy

class BNBConfig(QuantizationConfig):
    """Config class for BNB.

    Reference:
    """

    def __init__(
        self,
        weight_bits: int = 4,
        blocksize: int = 64,
        quant_type: str = "nf4",
        quant_storage: bool = torch.uint8,
        compress_statistics: bool = False,
    ) -> None:
        self.weight_bits = weight_bits
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.quant_storage = quant_storage
        self.compress_statistics = compress_statistics
        self.quant_map = torch.tensor([-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0000,
                                        0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000]).to(torch.cuda.current_device())

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"BNB, but got {self.weight_bits} bits.")
        if self.quant_storage != torch.uint8:
            raise ValueError(
                "Currently, only 8-bit quantization storage is supported for "
                f"BNB, but got {self.quant_storage} bits.")
        if self.quant_type != "nf4":
            raise ValueError(
                "Currently, only nf4 quantization type is supported for "
                f"BNB, but got {self.quant_type} bits.")
            
        if self.quant_storage == torch.uint8:
            self.pack_factor = 2

    def __repr__(self) -> str:
        return (f"BNBConfig(weight_bits={self.weight_bits}, "
                f"blocksize={self.blocksize}, "
                f"quant_type={self.quant_type}, "
                f"quant_storage={self.quant_storage}, "
                f"compress_statistics={self.compress_statistics})")

    def get_name(self) -> str:
        return "bnb"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BNBConfig":
        weight_bits = cls.get_from_keys(config, ["weight_bits", "w_bit", "bits"])
        blocksize = cls.get_from_keys(config, ["blocksize"])
        quant_type = cls.get_from_keys(config, ["quant_type"])
        quant_storage = cls.get_from_keys(config, ["quant_storage"])
        if quant_storage == "uint8":
            quant_storage = torch.uint8
        compress_statistics = cls.get_from_keys(config, ["compress_statistics"])
        return cls(weight_bits, blocksize, quant_type, quant_storage, compress_statistics)

    def get_linear_method(self) -> "BNBLinearMethod":
        return BNBLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError


class BNBLinearMethod(LinearMethodBase):
    """Linear method for BNB.

    Args:
        quant_config: The BNB quantization config.
    """

    def __init__(self, quant_config: BNBConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size_per_partition % self.quant_config.blocksize != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                dtype=self.quant_config.quant_storage,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        absmax = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.blocksize,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            absmax, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.blocksize,
            })
        return {
            "weight": qweight,
            "absmax": absmax
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["weight"]
        absmax = weights["absmax"]
        
        # TODO: Init once.
        # if x.size(0) == 1:
        #     import pdb; pdb.set_trace()
        print("x.dtype: ", x.dtype)
        quant_state = QuantState(absmax.contiguous().view(-1), dtype=x.dtype)
        quant_state.shape = torch.Size([qweight.shape[0], qweight.shape[1] * self.quant_config.pack_factor])
        quant_state.blocksize = self.quant_config.blocksize
        quant_state.quant_type = self.quant_config.quant_type
        quant_state.code = self.quant_config.quant_map
        
        # pack_factor = self.quant_config.pack_factor
        # out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        # reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold        
        bias = None #if self.bias is None else self.bias.to(x.dtype)
        # FIXME: Shape mismatch when bs = 1.
        # import pdb; pdb.set_trace()
        if x.size(0) == 1:
            out = x @ bnb.functional.dequantize_4bit(qweight.contiguous().view(-1,1), quant_state=quant_state).t()
        else:
            out = bnb.matmul_4bit(x, qweight.contiguous().view(-1,1).t(), bias=bias, quant_state=quant_state)
        return out
        # return out.reshape(out_shape)


class BNBDORALinearMethod(LinearMethodBase):
    """Linear method for BNB.

    Args:
        quant_config: The BNB quantization config.
    """
    pass
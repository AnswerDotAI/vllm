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
import torch.nn.functional as F

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
        lora_rank: int = None,
        compute_dtype: torch.dtype = None
    ) -> None:
        self.weight_bits = weight_bits
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.quant_storage = quant_storage
        self.compress_statistics = compress_statistics
        self.quant_map = torch.tensor([-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0000,
                                        0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000]).to(torch.cuda.current_device())
        self.lora_rank = lora_rank
        self.compute_dtype = compute_dtype
        
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
                f"compress_statistics={self.compress_statistics}, "
                f"lora_rank={self.lora_rank}, "
                f"compute_dtype={self.compute_dtype})")

    def get_name(self) -> str:
        return "bnb"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [ torch.bfloat16, torch.float16, torch.float32]

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
        lora_rank = config.get("lora_rank", None)
        compute_dtype = config.get("compute_dtype", None)
        if compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        elif compute_dtype == "float16":
            compute_dtype = torch.float16
        elif compute_dtype == "float32":
            compute_dtype = torch.float32
        return cls(weight_bits, blocksize, quant_type, quant_storage, 
                   compress_statistics, lora_rank, compute_dtype)

    def get_linear_method(self) -> "BNBLinearMethod":
        if self.lora_rank is None:
            return BNBLinearMethod(self)
        else:
            return BNBDORALinearMethod(self)

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
        quant_state = QuantState(absmax.contiguous().view(-1), dtype=x.dtype)
        quant_state.shape = torch.Size([qweight.shape[0], qweight.shape[1] * self.quant_config.pack_factor])
        quant_state.blocksize = self.quant_config.blocksize
        quant_state.quant_type = self.quant_config.quant_type
        quant_state.code = self.quant_config.quant_map
        
        bias = None #if self.bias is None else self.bias.to(x.dtype)
        # FIXME: Shape mismatch when bs = 1.
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
                
        # Don't set weight attributes, won't be used in tensor parallelism.
        lora_A = Parameter(
            torch.empty(
                self.quant_config.lora_rank,
                input_size_per_partition,
                dtype=self.quant_config.compute_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            lora_A, {
                "input_dim": 1,
            })
        
        lora_B = Parameter(
            torch.empty(
                output_size_per_partition,
                self.quant_config.lora_rank,
                dtype=self.quant_config.compute_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            lora_B, {
                "output_dim": 0,
            })

        # precomputed m * || W + AB ||, where m is the magnitude parameter.
        rescale = Parameter(
            torch.empty(
                output_size_per_partition,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            rescale, {
                "output_dim": 0,
            })
        

        return {
            "weight": qweight,
            "absmax": absmax,
            "lora_A": lora_A,
            "lora_B": lora_B,
            "rescale": rescale
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["weight"]
        absmax = weights["absmax"]
        lora_A = weights["lora_A"]
        lora_B = weights["lora_B"]
        rescale = weights["rescale"]
        
        # TODO: Init once.
        quant_state = QuantState(absmax.contiguous().view(-1), dtype=x.dtype)
        quant_state.shape = torch.Size([qweight.shape[0], qweight.shape[1] * self.quant_config.pack_factor])
        quant_state.blocksize = self.quant_config.blocksize
        quant_state.quant_type = self.quant_config.quant_type
        quant_state.code = self.quant_config.quant_map
        
        bias = None #if self.bias is None else self.bias.to(x.dtype)
        
        # TODO: This is inefficient. Fused qlora kernel. (w, lora_a, lora_b, rescale)
        w = bnb.functional.dequantize_4bit(qweight.contiguous().view(-1,1), quant_state=quant_state)
        w = rescale.view(-1,1) * (w + lora_B @ lora_A)        
        return x @ w.t()

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
# from .triton_mm import triton_mixed_mm
# os.environ['TRITON_ALWAYS_COMPILE']="0"
import bitblas
from bitblas.cache import global_operator_cache
from bitblas.module import BITBLAS_TARGET, BITBLAS_DATABASE_PATH

BITBLAS_DATABASE_PATH = "/workspace/.cache/bitblas"

def _get_or_create_bitblas_operator(config):
	if global_operator_cache.size() == 0:
		global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

	bitblas_matmul = global_operator_cache.get(config)
	if bitblas_matmul is None:
		# should disable tuning for the first time because we may require loading bitblas operator from database.
		bitblas_matmul = bitblas.Matmul(config)
		bitblas_matmul.hardware_aware_finetune(topk=20)
		global_operator_cache.add(config, bitblas_matmul)
		global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
		print("BitBLAS Tuning done, appended operator to global_operator_cache.")
	else:
		print("BitBLAS Operator found in global_operator_cache.")
	return bitblas_matmul


class BitBlasConfig(QuantizationConfig):
    """Config class for ...
    """

    def __init__(
        self,
        group_size: int,
        nbits: Union[int, Dict],
        lora_rank: int = None,
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size
        self.lora_rank = lora_rank
        if self.group_size not in [32, 64, 128, 256]:
            raise ValueError(
                "Currently, only group sizes [32, 64, 128, 256] "
                "are supported for _weight_int4pack_mm_cuda, but got group_size of "
                f"{self.group_size}")
        
        # 4 Bits packed into 8 bit datatype
        self.nbits = nbits
        
        if isinstance(nbits, int):
            self.pack_factor = 8 // nbits
        elif isinstance(nbits, Dict):
            # {layer_name: nbits}
            self.pack_factor = {k: 8 // v for k,v in nbits.items()}
                
    def __repr__(self) -> str:
        return f"BitBlasConfig(group_size={self.group_size})"

    @classmethod
    def get_name(cls) -> str:
        return "bitblas"

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
    def from_config(cls, config: Dict[str, Any]) -> "BitBlasConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        nbits = cls.get_from_keys(config, ["nbits"])
        lora_rank = config.get("lora_rank", None)
        return cls(group_size, nbits, lora_rank)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["BitBlasLinearMethod"]:
        if isinstance(layer, LinearBase):
            if self.lora_rank is None:
                return BitBlasLinearMethod(self)
            else:
                return BitBlasDORALinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class BitBlasLinearMethod(LinearMethodBase):
    """Linear method for bitblas. Supporting mixed 4/2 bits.

    Args:
        quant_config: The bitblas quantization config.
    """
    
    def __init__(self, quant_config: BitBlasConfig):
        self.quant_config = quant_config
        self.BITBLAS_OPT_M = [1, 16, 32, 64, 128, 256, 512]

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
            self.layer_nbits = self.quant_config.nbits[layer_name]
            self.layer_pack_factor = self.quant_config.pack_factor
        elif isinstance(self.quant_config.nbits, Dict):
            # {layer_name: nbits}
            self.layer_nbits = self.quant_config.nbits
            self.layer_pack_factor = self.quant_config.pack_factor[layer_name]
        
        if self.layer_nbits == 4:
            W_dtype = "uint4"
        elif self.layer_nbits == 2:
            W_dtype = "uint2"
        else:
            raise ValueError(f"Unsupported nbits: {self.quant_config.nbits}")
        
        if params_dtype != torch.half:
            raise ValueError(
                f"The params dtype must be half, but got {params_dtype}")
        
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

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        zeros = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            zeros,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("zeros", zeros)
        set_weight_attrs(zeros, extra_weight_attrs)
        
        
        N = output_size_per_partition
        K = input_size_per_partition * self.layer_pack_factor
        print(f"Tuning BitBLAS for {layer_name} with nbits {self.layer_nbits}-bit {K}x{N}")
        self.matmul_config = bitblas.MatmulConfig(M=self.BITBLAS_OPT_M,
                                                    N=N,
                                                    K=K,
                                                    A_dtype="float16",  
                                                    W_dtype=W_dtype,  
                                                    accum_dtype="float16",  
                                                    out_dtype="float16",  
                                                    layout="nt",  
                                                    with_bias=False, 
                                                    group_size=self.quant_config.group_size,
                                                    with_scaling=True,  
                                                    with_zeros=True,  
                                                    zeros_mode="original",  
                                                    #fast_decoding=True
                                                    )
        self.matmul_eng = _get_or_create_bitblas_operator(self.matmul_config)
        

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        qweight = layer.qweight
        scales = layer.scales
        zeros = layer.zeros
        output_size = qweight.size(0) * self.layer_pack_factor
        
        
        # x : bs x seq_len x hidden_size
        origin_x_size = x.size()
        new_shape = origin_x_size[:-1] + (output_size,)
        x_reshaped = x.reshape(-1, origin_x_size[-1])
        
        output = self.matmul_eng(x_reshaped, qweight, scales, zeros)

        output = output.reshape(new_shape)
        
        if bias is not None:
            output.add_(bias)  # In-place add

        return output


import torch._dynamo
torch._dynamo.config.suppress_errors = True

class BitBlasDORALinearMethod(LinearMethodBase):
    """Linear method for bitblas. Supporting mixed 4/2 bits.

    Args:
        quant_config: The bitblas quantization config.
    """
    
    def __init__(self, quant_config: BitBlasConfig):
        self.quant_config = quant_config
        self.BITBLAS_OPT_M = [1, 16, 32, 64, 128, 256, 512]

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
            self.layer_pack_factor = self.pack_factor
        elif isinstance(self.quant_config.nbits, Dict):
            # {layer_name: nbits}
            self.layer_pack_factor = self.pack_factor[layer_name]        

        if params_dtype != torch.half:
            raise ValueError(
                f"The params dtype must be half, but got {params_dtype}")
        
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

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        zeros = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            zeros,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("zeros", zeros)
        set_weight_attrs(zeros, extra_weight_attrs)
        
        if isinstance(self.quant_config.nbits, dict):
            layer_nbits = self.quant_config.nbits[layer_name]
        elif isinstance(self.quant_config.nbits, int):
            layer_nbits = self.quant_config.nbits
            
        if layer_nbits == 4:
            W_dtype = "uint4"
        elif layer_nbits == 2:
            W_dtype = "uint2"
        else:
            raise ValueError(f"Unsupported nbits: {self.quant_config.nbits}")
        
        N = output_size_per_partition
        K = input_size_per_partition * self.layer_pack_factor
        print(f"Tuning BitBLAS for {layer_name} with nbits {layer_nbits}-bit {K}x{N}")
        self.matmul_config = bitblas.MatmulConfig(M=self.BITBLAS_OPT_M,
                                                    N=N,
                                                    K=K,
                                                    A_dtype="float16",  
                                                    W_dtype=W_dtype,  
                                                    accum_dtype="float16",  
                                                    out_dtype="float16",  
                                                    layout="nt",  
                                                    with_bias=False, 
                                                    group_size=self.quant_config.group_size,
                                                    with_scaling=True,  
                                                    with_zeros=True,  
                                                    zeros_mode="original",  
                                                    #fast_decoding=True
                                                    )
        self.matmul_eng = _get_or_create_bitblas_operator(self.matmul_config)
        
                
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
        
    # def dora_layer(self, x, output, rescale, lora_A, lora_B):
    #     # Use in-place operations where possible
    #     x_lora = torch.mm(x, lora_A.t())
    #     x_lora = torch.mm(x_lora, lora_B.t())
    #     output.add_(x_lora)
    #     output.mul_(rescale.view(1, -1))
    #     return output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        zeros = layer.zeros
        output_size = qweight.size(0) * self.quant_config.pack_factor
        lora_A, lora_B, rescale = layer.lora_A, layer.lora_B, layer.rescale
        
        
        # x : bs x seq_len x hidden_size
        origin_x_size = x.size()
        new_shape = origin_x_size[:-1] + (output_size,)
        x_reshaped = x.reshape(-1, origin_x_size[-1])
        
        output = self.matmul_eng(x_reshaped, qweight, scales, zeros)

        output = output.reshape(new_shape)

        
        # rescale 
        # output = rescale.view(1,-1) * (output + x @ lora_A.t() @ lora_B.t()) 
        output = self.dora_layer(x, output, rescale, lora_A, lora_B)
        output = output.reshape(new_shape)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

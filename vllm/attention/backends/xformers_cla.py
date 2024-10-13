"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType, 
                                              SharedSelfAttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersCLABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "xformers"

    @staticmethod
    def get_impl_cls() -> Type["XFormersCLAImpl"]:
        return XFormersCLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return XFormersCLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["XFormersCLAMetadataBuilder"]:
        return XFormersCLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersCLAMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersCLA backend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Number of query tokens for each request in the batch.
    # Currently, we require that all requests have the same number of query
    # tokens during the decoding phase. When speculavie decoding is enabled,
    # decode_query_len might be greater than 1. In all other cases, it is 1.
    decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersCLAMetadata"] = None
    _cached_decode_metadata: Optional["XFormersCLAMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None
    
    # Debug Shared KV Cache (Cross Layer Attention)
    shared_self_attention_types: Optional[List[SharedSelfAttentionType]] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return ((self.encoder_seq_lens is not None)
                and (self.encoder_seq_lens_tensor is not None)
                and (self.max_encoder_seq_len is not None))

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return (self.is_all_encoder_attn_metadata_set
                and (self.cross_slot_mapping is not None)
                and (self.cross_block_tables is not None))

    @property
    def prefill_metadata(self) -> Optional["XFormersCLAMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersCLAMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        self._cached_prefill_metadata.shared_self_attention_types = []
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersCLAMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersCLAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        self._cached_decode_metadata.shared_self_attention_types = []
        return self._cached_decode_metadata


def _get_attn_bias(
    attn_metadata: XFormersCLAMetadata,
    attn_type: AttentionType,
) -> Optional[AttentionBias]:
    '''
    Extract appropriate attention bias from attention metadata
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:
    * Appropriate attention bias value given the attention type
    '''

    if attn_type == AttentionType.DECODER:
        return attn_metadata.attn_bias
    elif attn_type == AttentionType.ENCODER:
        return attn_metadata.encoder_attn_bias
    else:
        # attn_type == AttentionType.ENCODER_DECODER
        return attn_metadata.cross_attn_bias


def _set_attn_bias(
    attn_metadata: XFormersCLAMetadata,
    attn_bias: List[Optional[AttentionBias]],
    attn_type: AttentionType,
) -> None:
    '''
    Update appropriate attention bias field of attention metadata,
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_bias: The desired attention bias value
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention
    '''

    if attn_type == AttentionType.DECODER:
        attn_metadata.attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER:
        attn_metadata.encoder_attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        attn_metadata.cross_attn_bias = attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _get_seq_len_block_table_args(
    attn_metadata: XFormersCLAMetadata,
    is_prompt: bool,
    attn_type: AttentionType,
) -> tuple:
    '''
    The particular choice of sequence-length- and block-table-related
    attributes which should be extracted from attn_metadata is dependent
    on the type of attention operation.

    Decoder attn -> select entirely decoder self-attention-related fields
    Encoder/decoder cross-attn -> select encoder sequence lengths & 
                                  cross-attn block-tables fields
    Encoder attn -> select encoder sequence lengths fields & no block tables
    
    Arguments:

    * attn_metadata: Attention metadata structure associated with attention op
    * is_prompt: True if prefill, False otherwise
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:

    * Appropriate sequence-lengths tensor
    * Appropriate max sequence-length scalar
    * Appropriate block tables (or None)
    '''

    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_lens_tensor, max_seq_len,
                attn_metadata.block_tables)
    elif attn_type == AttentionType.ENCODER_DECODER:
        # Enc/dec cross-attention KVs match encoder sequence length;
        # cross-attention utilizes special "cross" block tables
        return (attn_metadata.encoder_seq_lens_tensor,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.cross_block_tables)
    elif attn_type == AttentionType.ENCODER:
        # No block tables associated with encoder attention
        return (attn_metadata.encoder_seq_lens_tensor,
                attn_metadata.max_encoder_seq_len, None)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class XFormersCLAMetadataBuilder(CommonMetadataBuilder[XFormersCLAMetadata]):

    _metadata_cls = XFormersCLAMetadata


class XFormersCLAImpl(AttentionImpl[XFormersCLAMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "XFormersCLA does not support block-sparse attention.")
        if logits_soft_cap is not None:
            raise ValueError(
                "XFormersCLA does not support attention logits soft capping.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "XFormersCLAMetadata",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
        compute_new_kv: bool = True,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * XFormersImpl.forward() may be invoked for both self- and cross-
          attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode
        
        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:
    
            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len)
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)
    
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Check that appropriate attention metadata attributes are
        # selected for the desired attention type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        query = query.view(-1, self.num_heads, self.head_size)
        if compute_new_kv:
            if key is not None:
                assert value is not None
                key = key.view(-1, self.num_kv_heads, self.head_size)
                value = value.view(-1, self.num_kv_heads, self.head_size)
            else:
                assert value is None
        else:
            assert key is None and value is None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                # if compute_new_kv:
                PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                    value_cache,
                                                    updated_slot_mapping,
                                                    self.kv_cache_dtype,
                                                    k_scale, v_scale)
                # print("Writing to KV cache")
                # print(f"updated_slot_mapping: {updated_slot_mapping}")
                # print(f"key.shape:{key.shape}, value.shape: {value.shape}")
                # else:
                #     # Use existing KV cache in proceeding layers sharing the cache.
                #     # query: shape = [num_tokens, num_heads * head_size]
                #     # key: shape = [num_tokens, num_kv_heads * head_size]
                #     # value: shape = [num_tokens, num_kv_heads * head_size]
                #     # kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                
                #     # For decoding KV is not needed, KV cache is used instead - so no changes needed.
                #     # For prefill, KV is needed, so we need to extract it from the cache.
                #     # Cached prefill is implemented in PagedAttention.forward_prefix() as a new triton kernel.
                #     # Uses KV from cache, to compute self-attention.
                #     # print("Re-using KV cache")                    
                #     pass

        if attn_type != AttentionType.ENCODER:
            # Decoder self-attention supports chunked prefill.
            # Encoder/decoder cross-attention requires no chunked
            # prefill (100% prefill or 100% decode tokens, no mix)
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
        else:
            # Encoder attention - chunked prefill is not applicable;
            # derive token-count from query shape & and treat them
            # as 100% prefill tokens
            assert attn_metadata.num_encoder_tokens is not None
            num_prefill_tokens = attn_metadata.num_encoder_tokens
            num_decode_tokens = 0

        if attn_type == AttentionType.DECODER:
            # Only enforce this shape-constraint for decoder
            # self-attention
            if compute_new_kv:
                assert key.shape[0] == num_prefill_tokens + num_decode_tokens
                assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                if compute_new_kv:
                    # print("Prefilling")
                    # print(prefill_meta.block_tables.shape, kv_cache.shape if kv_cache is not None else 'None')
                    # print(prefill_meta.block_tables)
                    # normal attention.
                    # block tables are empty if the prompt does not have a cached
                    # prefix.
                    prefill_meta.shared_self_attention_types.append(SharedSelfAttentionType.PREFILL_KV_NEW.name)
                elif kv_cache.numel() > 0:
                    # print("Prefilling with KV cache")
                    # Split kv_cache into key_cache and value_cache.
                    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
                    block_size = value_cache.size(-1)
                    
                    # Extract KV from cache using slot_mapping.
                    k_reshaped, v_reshaped = convert_cache_to_original_shape(key_cache, value_cache, self.num_kv_heads, self.head_size)
                    slot_mapping = prefill_meta.slot_mapping
                    block_idxs, token_idxs = slot_mapping // block_size, slot_mapping % block_size
                    key, value = k_reshaped[block_idxs, token_idxs], v_reshaped[block_idxs, token_idxs]
                    if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
                        key = (key.view(torch.float8_e4m3fn).to(torch.float32) * k_scale).to(query.dtype)
                        value = (value.view(torch.float8_e4m3fn).to(torch.float32) * v_scale).to(query.dtype)
                    elif self.kv_cache_dtype == "fp8_e5m2":
                        key = (key.view(torch.float8_e5m2).to(torch.float32) * k_scale).to(query.dtype)
                        value = (value.view(torch.float8_e5m2).to(torch.float32) * v_scale).to(query.dtype)
                    
                    prefill_meta.shared_self_attention_types.append(SharedSelfAttentionType.PREFILL_KV_SHARED.name)
                else:
                    # print("Profiling")
                    if not compute_new_kv:
                        # kv_cache is not created yet during profiling, so we need to create key and value.
                        key = torch.zeros(query.size(0), self.num_kv_heads, self.head_size, dtype=query.dtype, device=query.device)
                        value = torch.zeros(query.size(0), self.num_kv_heads, self.head_size, dtype=query.dtype, device=query.device)
                    prefill_meta.shared_self_attention_types.append(SharedSelfAttentionType.PREFILL_PROFILING.name)

                out = self._run_memory_efficient_xformers_forward(query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_tokens].shape
                output[:num_prefill_tokens] = out
            else:
                # TODO: Test this KV Cache Sharing. 
                # print("Prefilling with KV cache (prefix-cache enabled)")
                # print(prefill_meta.block_tables.shape, kv_cache.shape if kv_cache is not None else 'None')
                # print(prefill_meta.block_tables)
                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None
                
                if kv_cache.numel() > 0 and not compute_new_kv:
                    # Split kv_cache into key_cache and value_cache.
                    key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
                    block_size = value_cache.size(-1)
                    
                    # Extract KV from cache using slot_mapping.
                    k_reshaped, v_reshaped = convert_cache_to_original_shape(key_cache, value_cache, self.num_kv_heads, self.head_size)
                    slot_mapping = prefill_meta.slot_mapping
                    block_idxs, token_idxs = slot_mapping // block_size, slot_mapping % block_size
                    key, value = k_reshaped[block_idxs, token_idxs], v_reshaped[block_idxs, token_idxs]
                    if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
                        key = (key.view(torch.float8_e4m3fn).to(torch.float32) * k_scale).to(query.dtype)
                        value = (value.view(torch.float8_e4m3fn).to(torch.float32) * v_scale).to(query.dtype)
                    elif self.kv_cache_dtype == "fp8_e5m2":
                        key = (key.view(torch.float8_e5m2).to(torch.float32) * k_scale).to(query.dtype)
                        value = (value.view(torch.float8_e5m2).to(torch.float32) * v_scale).to(query.dtype)

                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                # print(f"query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}")
                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    self.kv_cache_dtype,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.query_start_loc,
                    prefill_meta.seq_lens_tensor,
                    prefill_meta.context_lens_tensor,
                    prefill_meta.max_query_len,
                    self.alibi_slopes,
                    self.sliding_window,
                    k_scale,
                    v_scale,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
                
                if compute_new_kv:
                    prefill_meta.shared_self_attention_types.append(SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV.name)
                else:
                    prefill_meta.shared_self_attention_types.append(SharedSelfAttentionType.PREFILL_PREFIX_CACHED_KV_SHARED.name)
                
                
        if decode_meta := attn_metadata.decode_metadata:
            # print("Decoding")
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = _get_seq_len_block_table_args(decode_meta, False, attn_type)

            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )
            if compute_new_kv:
                decode_meta.shared_self_attention_types.append(SharedSelfAttentionType.DECODE_KV_NEW.name)
            else:
                decode_meta.shared_self_attention_types.append(SharedSelfAttentionType.DECODE_KV_SHARED.name)

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersCLAMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        """

        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        attn_bias = _get_attn_bias(attn_metadata, attn_type)
        if attn_bias is None:
            if self.alibi_slopes is None:
                if (attn_type == AttentionType.ENCODER_DECODER):
                    assert attn_metadata.seq_lens is not None
                    assert attn_metadata.encoder_seq_lens is not None

                    # Default enc/dec cross-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens, attn_metadata.encoder_seq_lens)
                elif attn_type == AttentionType.ENCODER:
                    assert attn_metadata.encoder_seq_lens is not None

                    # Default encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.encoder_seq_lens)
                else:
                    assert attn_metadata.seq_lens is not None

                    # Default decoder self-attention mask is causal
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        attn_metadata.seq_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_bias = [attn_bias]
            else:
                assert attn_metadata.seq_lens is not None
                attn_bias = _make_alibi_bias(self.alibi_slopes,
                                             self.num_kv_heads, query.dtype,
                                             attn_metadata.seq_lens)

            _set_attn_bias(attn_metadata, attn_bias, attn_type)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_bias[0],
                p=0.0,
                scale=self.scale)
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases


def convert_cache_to_original_shape(k_cache, v_cache, num_kv_heads, head_size):
    """
    Convert k_cache and v_cache from 5D and 4D shapes back to 
    (num_blocks, block_size, num_kv_heads, head_size)
    
    Args:
    k_cache (torch.Tensor): Key cache of shape [num_blocks, num_heads, head_size/x, block_size, x]
    v_cache (torch.Tensor): Value cache of shape [num_blocks, num_heads, head_size, block_size]
    num_kv_heads (int): Number of key/value heads
    head_size (int): Size of each head
    
    Returns:
    tuple: (k_standard, v_standard) - Reshaped key and value caches
    """
    num_blocks, _, _, block_size, x = k_cache.shape
    
    # Reshape key cache
    k_standard = k_cache.permute(0, 3, 1, 2, 4)
    k_standard = k_standard.reshape(num_blocks, block_size, num_kv_heads, head_size)
    
    # Reshape value cache
    v_standard = v_cache.permute(0, 3, 1, 2)
    v_standard = v_standard.reshape(num_blocks, block_size, num_kv_heads, head_size)
    
    return k_standard, v_standard

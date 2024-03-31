from typing import cast

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module, LayerNorm
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter

from timm.models.vision_transformer import Block

class CompressedBlock(Block):
    def forward(self, x):
        # TODO: implement orthogonal transformations
        # if self.attn_shortcut_Q is not None:
        #     rotated_residual = matmul(residual, self.attn_shortcut_Q)
        #     hidden_states = rotated_residual + hidden_states
        # else:
        #     hidden_states = residual + hidden_states

        # if self.mlp_shortcut_Q is not None:
        #     rotated_residual = matmul(residual, self.mlp_shortcut_Q)
        #     hidden_states = rotated_residual + hidden_states
        # else:
        #     hidden_states = residual + hidden_states

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class VitLayerAdapter(LayerAdapter):
    def __init__(self, layer) -> None:
        super().__init__()
        self._layer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        raise NotImplementedError

    @property
    def hidden_states_output_position(self) -> int:
        raise NotImplementedError

    def get_first_layernorm(self) -> Module:
        return self.layer.norm1

    def get_second_layernorm(self) -> Module:
        return self.layer.norm2

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.attn.qkv]

    def get_attention_output(self) -> Linear:
        return self.layer.attn.proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.fc1]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.fc2


class VitModelAdapter(ModelAdapter):
    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        raise NotImplementedError

    @property
    def config_type(self) -> type:
        raise NotImplementedError

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        return self.model.embed_dim

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return True

    @property
    def original_layer_type(self) -> type:
        raise NotImplementedError

    @property
    def original_layer_norm_type(self) -> type:
        return LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        return VitLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        raise NotImplementedError

    @property
    def use_cache(self) -> bool:
        raise NotImplementedError

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        raise NotImplementedError

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        raise NotImplementedError

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        raise NotImplementedError

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.blocks]

    def get_raw_layer_at(self, index: int) -> Module:
        raise NotImplementedError

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        raise NotImplementedError

    def get_embeddings(self) -> list[Module]:
        return [self.model.cls_token]

    def get_pre_head_layernorm(self) -> type:
        return self.model.norm

    def get_lm_head(self) -> Linear:
        return self.model.head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        raise NotImplementedError

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        raise NotImplementedError

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        raise NotImplementedError

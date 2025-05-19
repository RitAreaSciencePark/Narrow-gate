from typing import Any, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from .configuration_aimv2 import AIMv2Config
from flax.core import frozen_dict
from transformers import FlaxPreTrainedModel
from transformers.modeling_flax_outputs import FlaxBaseModelOutput

__all__ = ["FlaxAIMv2Model"]


class FlaxRMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones_init(), (dim,))
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        output = output * scale.astype(x.dtype)
        return output

    def _norm(self, x: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps)


class FlaxAIMv2SwiGLUFFN(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_features = self.config.intermediate_size
        in_features = self.config.hidden_size
        bias = self.config.use_bias

        x1 = nn.Dense(hidden_features, use_bias=bias, dtype=self.dtype, name="fc1")(x)
        x2 = nn.Dense(hidden_features, use_bias=bias, dtype=self.dtype, name="fc3")(x)
        x = nn.silu(x1) * x2
        x = nn.Dense(in_features, use_bias=bias, dtype=self.dtype, name="fc2")(x)
        return x


class FlaxAIMv2PatchEmbed(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        patch_size = (self.config.patch_size, self.config.patch_size)
        x = x.transpose(0, 2, 3, 1)  # (N C H W) -> (N H W C)
        x = nn.Conv(
            self.config.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding=(0, 0),
            dtype=self.dtype,
            name="proj",
        )(x)
        x = jax.lax.collapse(x, 1, 3)  # (N, H * W, F)
        x = FlaxRMSNorm(self.config.rms_norm_eps, name="norm")(x)
        return x


class FlaxAIMv2ViTPreprocessor(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tokens = FlaxAIMv2PatchEmbed(self.config, dtype=self.dtype, name="patchifier")(
            x
        )
        _, N, _ = tokens.shape
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_patches, self.config.hidden_size),
        )
        tokens = tokens + pos_embed[:, :N].astype(tokens.dtype)
        return tokens

    @property
    def num_patches(self) -> int:
        return (self.config.image_size // self.config.patch_size) ** 2


class FlaxAIMv2Attention(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        B, N, C = x.shape
        dim, num_heads = self.config.hidden_size, self.config.num_attention_heads

        qkv = nn.Dense(
            dim * 3, use_bias=self.config.qkv_bias, dtype=self.dtype, name="qkv"
        )(x)
        qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = nn.dot_product_attention_weights(
            q.swapaxes(-3, -2),  # [B, N, H, C]
            k.swapaxes(-3, -2),
            mask=mask,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        attn_weights = nn.Dropout(
            self.config.attention_dropout, deterministic=deterministic, name="attn_drop"
        )(attn_weights)

        x = (attn_weights @ v).swapaxes(1, 2).reshape(B, N, C)
        x = nn.Dense(dim, use_bias=self.config.use_bias, dtype=self.dtype, name="proj")(
            x
        )
        x = nn.Dropout(
            self.config.projection_dropout,
            deterministic=deterministic,
            name="proj_drop",
        )(x)
        return (x, attn_weights) if output_attentions else (x, None)


class FlaxAIMv2Block(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attn = FlaxAIMv2Attention(self.config, dtype=self.dtype, name="attn")
        self.norm_1 = FlaxRMSNorm(self.config.rms_norm_eps, name="norm_1")
        self.mlp = FlaxAIMv2SwiGLUFFN(self.config, dtype=self.dtype, name="mlp")
        self.norm_2 = FlaxRMSNorm(self.config.rms_norm_eps, name="norm_2")

    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        features, attention = self.attn(
            self.norm_1(x),
            mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        x = x + features
        x = x + self.mlp(self.norm_2(x))
        return x, attention


class FlaxAIMv2Transformer(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[
        jax.Array, Optional[Tuple[jax.Array, ...]], Optional[Tuple[jax.Array, ...]]
    ]:
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for blk_id, block in enumerate(range(self.config.num_hidden_layers)):
            tokens, attention = FlaxAIMv2Block(
                self.config, dtype=self.dtype, name=f"layers_{blk_id}"
            )(
                tokens,
                mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                hidden_states += (tokens,)
            if output_attentions:
                attentions += (attention,)
        tokens = FlaxRMSNorm(self.config.rms_norm_eps, name="post_trunk_norm")(tokens)
        return tokens, hidden_states, attentions


class FlaxAIMv2Module(nn.Module):
    config: AIMv2Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[
        jax.Array, Optional[Tuple[jax.Array, ...]], Optional[Tuple[jax.Array, ...]]
    ]:
        x = FlaxAIMv2ViTPreprocessor(
            self.config, dtype=self.dtype, name="preprocessor"
        )(x)
        x, hidden_states, attentions = FlaxAIMv2Transformer(
            self.config, dtype=self.dtype, name="trunk"
        )(
            x,
            mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return x, hidden_states, attentions


class FlaxAIMv2PretrainedModel(FlaxPreTrainedModel):
    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: AIMv2Config,
        input_shape: Optional[Tuple[int, int, int, int]] = None,  # [B, C, H, W]
        dtype: jnp.dtype = jnp.float32,
        **kwargs: Any,
    ):
        if input_shape is None:
            input_shape = (1, 3, config.image_size, config.image_size)
        super().__init__(
            config,
            module=FlaxAIMv2Module(config, dtype=dtype),
            input_shape=input_shape,
            dtype=dtype,
            **kwargs,
        )

    def init_weights(
        self,
        rng: jax.Array,
        input_shape: Tuple[int, ...],
        params: Optional[frozen_dict.FrozenDict] = None,
    ) -> frozen_dict.FrozenDict:
        del params
        input_pixels = jnp.empty(input_shape)
        params = self.module.init(rng, input_pixels, deterministic=True)
        return params["params"]


class FlaxAIMv2Model(FlaxAIMv2PretrainedModel):
    def __call__(
        self,
        pixel_values: jax.Array,
        params: Optional[frozen_dict.FrozenDict] = None,
        mask: Optional[jax.Array] = None,
        dropout_rng: Optional[jax.Array] = None,
        deterministic: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[jax.Array],
        Tuple[jax.Array, Tuple[jax.Array, ...]],
        Tuple[jax.Array, Tuple[jax.Array, ...], Tuple[jax.Array, ...]],
        FlaxBaseModelOutput,
    ]:
        if params is None:
            params = self.params
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        rngs = None if deterministic else {"dropout": dropout_rng}

        x, hidden_states, attentions = self.module.apply(
            {"params": params},
            pixel_values,
            mask,
            rngs=rngs,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            res += (attentions,) if output_attentions else ()
            return res

        return FlaxBaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states,
            attentions=attentions,
        )


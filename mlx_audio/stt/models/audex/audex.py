# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Nemotron-Labs-Audex for mlx-audio STT.

Architecture (from nvidia/Nemotron-Labs-Audex-2B checkpoint_folder_full):
  NV-Whisper encoder (== Qwen2AudioEncoder, 128-mel, 30s clips -> 750 tokens/clip)
  -> RMSNorm + fc1(1280->4096) + relu^2 + fc2(4096->hidden) projector
  -> Nemotron-Dense decoder (RMSNorm, GQA, relu^2 MLP, RoPE theta 1e8)
Audio embeddings are spliced into <so_embedding> (id 29) positions of a ChatML
prompt, mirroring the reference inference_scripts_hf pipeline exactly.
"""

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig
from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoder


@dataclass
class ModelConfig:
    model_type: str = "audex"
    audio_config: EncoderConfig = None
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-5
    vocab_size: int = 205312
    rope_parameters: dict = field(default_factory=lambda: {"rope_theta": 100000000})
    audio_encoder_hidden_size: int = 1280
    audio_projector_intermediate_size: int = 4096
    audio_projector_norm_eps: float = 1e-5
    sound_token_id: int = 29
    sound_start_token: str = "<so_start>"
    sound_end_token: str = "<so_end>"
    sound_token: str = "<so_embedding>"
    sound_embedding_size: int = 750
    sound_clip_duration: float = 30.0
    eos_token_id: int = 11

    def __post_init__(self):
        if isinstance(self.audio_config, dict):
            self.audio_config = EncoderConfig.from_dict(self.audio_config)
        elif self.audio_config is None:
            self.audio_config = EncoderConfig()

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        )


class AudexProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.audio_encoder_hidden_size, eps=config.audio_projector_norm_eps)
        self.fc1 = nn.Linear(
            config.audio_encoder_hidden_size, config.audio_projector_intermediate_size, bias=False
        )
        self.fc2 = nn.Linear(
            config.audio_projector_intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.relu(self.fc1(self.norm(x))).square())


class NemotronAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=float(config.rope_parameters.get("rope_theta", 100000000)),
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class NemotronMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.relu(self.up_proj(x)).square())


class NemotronLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = NemotronAttention(config)
        self.mlp = NemotronMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class NemotronModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NemotronLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = Qwen2AudioEncoder(config.audio_config)
        self.audio_projector = AudexProjector(config)
        self.model = NemotronModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._tokenizer = None
        self._feature_extractor = None

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.model.layers))]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.model.embed_tokens(input_ids)
        if h.ndim == 2:
            h = h[None]
        if cache is None:
            cache = [None] * len(self.model.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.model.layers, cache):
            h = layer(h, mask, cache=c)
        return self.lm_head(self.model.norm(h))

    # ── audio front-end (mirrors reference audio_utils.py) ──
    def _split_clips(self, audio: np.ndarray) -> List[np.ndarray]:
        sr = 16000
        clip_samples = int(round(sr * self.config.sound_clip_duration))
        audio = audio.astype(np.float32)
        m = float(np.abs(audio).max()) if audio.size else 0.0
        if m > 1.0:
            audio = audio / m
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        n = max(1, int(np.ceil(audio.shape[0] / clip_samples)))
        clips = []
        for i in range(n):
            c = audio[i * clip_samples : (i + 1) * clip_samples]
            if c.shape[0] < clip_samples:
                c = np.pad(c, (0, clip_samples - c.shape[0]))
            clips.append(c)
        return clips

    def encode_audio(self, audio: np.ndarray) -> mx.array:
        """Waveform -> (1, clips*750, hidden) projected sound embeddings."""
        clips = self._split_clips(audio)
        feats = self._feature_extractor(
            clips, sampling_rate=16000, return_tensors="np",
            padding="max_length", return_attention_mask=False,
        ).input_features  # (clips, 128, 3000)
        dtype = self.audio_encoder.conv1.weight.dtype
        encoded = self.audio_encoder(mx.array(feats, dtype=dtype))  # (clips, 750, 1280)
        projected = self.audio_projector(encoded)  # (clips, 750, hidden)
        return projected.reshape(1, -1, projected.shape[-1])

    def _build_prompt(self, num_sound_tokens: int, user_prompt: str) -> mx.array:
        c = self.config
        sound = c.sound_start_token + c.sound_token * num_sound_tokens + c.sound_end_token
        # Matches reference build_prompt_template (non-reasoning): sound first.
        text = (
            f"<|im_start|>user\n{sound}\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>"
        )
        return mx.array(self._tokenizer.encode(text, add_special_tokens=False))

    def get_input_embeddings(self, audio, prompt: Optional[str] = None):
        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            audio = np.array(load_audio(audio), dtype=np.float32)
        elif isinstance(audio, mx.array):
            audio = np.array(audio, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        sound_embeds = self.encode_audio(audio)  # (1, N, hidden)
        prompt_ids = self._build_prompt(
            sound_embeds.shape[1], prompt or "Transcribe the speech in the input audio."
        )

        is_sound = prompt_ids == self.config.sound_token_id
        text_embeds = self.model.embed_tokens(mx.where(is_sound, 0, prompt_ids)[None])
        sound_embeds = sound_embeds.astype(text_embeds.dtype)

        idx = mx.clip(mx.cumsum(is_sound.astype(mx.int32)) - 1, 0, sound_embeds.shape[1] - 1)
        spliced = mx.where(
            is_sound[None, :, None], sound_embeds[:, idx, :], text_embeds
        )
        mx.eval(spliced)
        return prompt_ids, spliced, len(prompt_ids)

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        prompt: str = None,
        language: str = None,  # accepted for API parity; Audex is multilingual
        prefill_step_size: int = 2048,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        start = time.time()
        prompt_ids, embeds, n_prompt = self.get_input_embeddings(audio, prompt)
        prefill_time = time.time() - start

        eos_ids = {self.config.eos_token_id}
        im_end = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            eos_ids.add(im_end)

        sampler = make_sampler(temperature, top_p=top_p, top_k=top_k)
        tokens = []
        gen_start = time.time()
        for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            prefill_step_size=prefill_step_size,
        ):
            if token in eos_ids:
                break
            tokens.append(token)

        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        # Strip any reasoning residue defensively (reference split_thinking)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        text = text.strip()

        # Audex's ASR training bakes in a wrapper:
        #   Language: English. The spoken content of the audio is '...'
        # Unwrap it so callers get a bare transcript.
        import re

        detected_lang = None
        m = re.match(
            r"^Language:\s*([A-Za-z ]+?)\.\s*The spoken content of the audio is\s*'(.*)'\.?\s*$",
            text,
            re.DOTALL,
        )
        if m:
            detected_lang = m.group(1).strip().lower()
            text = m.group(2).strip()

        elapsed = time.time() - start
        gen_time = time.time() - gen_start
        return STTOutput(
            text=text,
            language=detected_lang or language,
            segments=[{"start": 0.0, "end": elapsed, "text": text}],
            prompt_tokens=n_prompt,
            generation_tokens=len(tokens),
            total_tokens=n_prompt + len(tokens),
            total_time=elapsed,
            prompt_tps=n_prompt / prefill_time if prefill_time > 0 else 0,
            generation_tps=len(tokens) / gen_time if gen_time > 0 else 0,
        )

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (p.startswith("audio_encoder") or p.startswith("audio_projector"))

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            if "embed_positions" in k:
                continue  # fixed sinusoids, computed at init
            if k.startswith("audio_encoder.conv") and k.endswith("weight") and v.ndim == 3:
                v = v.transpose(0, 2, 1)  # PyTorch (out,in,kW) -> MLX (out,kW,in)
            out[k] = v
        return out

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            pre = Path(model_path) / "audio_preprocessor"
            model._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                str(pre if pre.exists() else model_path)
            )
        finally:
            transformers.logging.set_verbosity(prev)

        enc = model.audio_encoder
        enc._embed_positions = enc._embed_positions.astype(enc.conv1.weight.dtype)
        return model

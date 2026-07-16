# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Nemotron-Labs-Audex text-to-speech for mlx-audio.

Pipeline (from nvidia/Nemotron-Labs-Audex-2B run_audio_gen_vllm.py):
  ChatML prompt "<|text to speech|> Generate speech for this transcription. {text}"
  -> Nemotron-Dense LM generates <speechcodec_N> tokens (single codebook, 50 fps)
     with classifier-free guidance (cond/null prompt pair, shared sampled tokens)
  -> Audex causal speech decoder (FSQ embedder + 12-layer Vocos transformer,
     4-frame lookahead depthwise conv, tanh patch head, 320 samples/frame)
  -> 16 kHz waveform.

The LM reuses the NemotronModel from the STT audex architecture; the model dir
holds the (symlinked) LM shards plus the speech decoder weights, with the
decoder config nested under config.json "speech_decoder".
"""

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.audex.audex import NemotronModel
from mlx_audio.stt.models.audex.audex import ModelConfig as LMConfig
from mlx_audio.tts.models.base import GenerationResult

SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools."
)
CODEC_FPS = 50


@dataclass
class ModelConfig:
    model_type: str = "audex_tts"
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    norm_eps: float = 1e-5
    vocab_size: int = 205312
    rope_parameters: dict = field(default_factory=lambda: {"rope_theta": 100000000})
    eos_token_id: int = 11
    speech_decoder: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        )

    def lm_config(self) -> LMConfig:
        return LMConfig.from_dict(self.__dict__)


# ── Audex causal speech decoder (port of modeling_audex_causal_speech_decoder.py) ──


class FSQTokenEmbedder(nn.Module):
    """Token id -> base-L digits scaled to [-1, 1] -> linear projection."""

    def __init__(self, output_dim: int, codebook_levels: List[int]):
        super().__init__()
        self.levels = codebook_levels
        basis, acc = [], 1
        for lvl in codebook_levels:
            basis.append(acc)
            acc *= lvl
        self._basis = mx.array(basis, dtype=mx.int64)
        self._levels = mx.array(codebook_levels, dtype=mx.int64)
        self.project_out = nn.Linear(len(codebook_levels), output_dim, bias=True)

    def __call__(self, indices: mx.array) -> mx.array:
        # indices: (B, T) int -> (B, T, len(levels))
        digits = (indices[..., None] // self._basis) % self._levels
        codes = digits.astype(mx.float32) * (
            2.0 / (self._levels.astype(mx.float32) - 1.0)
        ) - 1.0
        return self.project_out(codes.astype(self.project_out.weight.dtype))


class DecoderAttention(nn.Module):
    """Fused-QKV causal attention with traditional (paired) RoPE on full head_dim."""

    def __init__(self, dim: int, n_heads: int, rope_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(rope_dim, traditional=True, base=10000)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        B, L, D = x.shape
        qkv = self.c_attn(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q = self.rope(q)
        k = self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(B, L, D))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope_dim: int):
        super().__init__()
        self.att_norm = nn.RMSNorm(dim, eps=1e-6)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.att = DecoderAttention(dim, n_heads, rope_dim)
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        x = x + self.att(self.att_norm(x), mask)
        return x + self.fc2(nn.silu(self.fc1(self.ffn_norm(x))))


class SpeechDecoder(nn.Module):
    """Non-streaming port: decodes the full codec-token sequence in one pass."""

    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg.get("hidden_dim", 2048)
        self.hop_length = cfg.get("hop_length", 320)
        self.lookahead_steps = cfg.get("lookahead_steps", 4)
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.embedder = FSQTokenEmbedder(
            cfg.get("vq_dim", 2048), cfg.get("codebook_levels", [4] * 8)
        )
        self.fc_post_a = nn.Linear(cfg.get("vq_dim", 2048), dim, bias=False)
        self.lookahead_conv = nn.Conv1d(
            dim, dim, kernel_size=self.lookahead_steps + 1, groups=dim, bias=False
        )
        self.lookahead_proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.blocks = [
            DecoderBlock(dim, cfg.get("heads", 32), cfg.get("pos_meb_dim", 64))
            for _ in range(cfg.get("depth", 12))
        ]
        self.final_layer_norm = nn.RMSNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, self.hop_length, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        # tokens: (B, T) codec ids -> (B, T * hop_length) waveform
        x = self.fc_post_a(self.embedder(tokens))
        # lookahead: right-pad, depthwise conv over time, residual add
        h = mx.pad(x, ((0, 0), (0, self.lookahead_steps), (0, 0)))
        h = self.lookahead_proj(nn.silu(self.lookahead_conv(h)))
        x = x + h
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_layer_norm(x)
        wav = mx.tanh(self.head(x))
        return wav.reshape(wav.shape[0], -1)


# ── TTS model ──


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = NemotronModel(config.lm_config())
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder = SpeechDecoder(config.speech_decoder or {})
        self.sample_rate = self.decoder.sample_rate
        self._tokenizer = None
        self._speech_base: Optional[int] = None
        self._speechgen_end: Optional[int] = None

    # LM forward over token embeddings with KV caches
    def _lm_step(self, ids: mx.array, caches) -> mx.array:
        from mlx_lm.models.base import create_attention_mask

        h = self.model.embed_tokens(ids)
        mask = create_attention_mask(h, caches[0])
        for layer, c in zip(self.model.layers, caches):
            h = layer(h, mask, cache=c)
        return self.lm_head(self.model.norm(h))

    def _prompts(self, text: str):
        def template(body: str) -> str:
            return (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<|text to speech|> Generate speech for this transcription. "
                f"{body}<|im_end|>\n"
                f"<|im_start|>assistant\n<think></think><speechgen_start>"
            )

        enc = lambda s: self._tokenizer.encode(s, add_special_tokens=False)
        cond = enc(template(text))
        # null prompt for CFG: <unk> run sized to match the cond prompt length
        target, base = len(cond), len(enc(template("")))
        n = max(1, target - base)
        for _ in range(64):
            null = enc(template("<unk>" * n))
            if len(null) == target:
                break
            n += 1 if len(null) < target else -1
            if n < 1:
                n = 1
                break
        return mx.array(cond), mx.array(enc(template("<unk>" * n)))

    def _codec_token_map(self):
        if self._speech_base is None:
            t = self._tokenizer
            self._speech_base = t.convert_tokens_to_ids("<speechcodec_0>")
            # verify contiguity so decode is arithmetic, not a 65k dict
            assert t.convert_tokens_to_ids("<speechcodec_1>") == self._speech_base + 1
            assert t.convert_tokens_to_ids("<speechcodec_100>") == self._speech_base + 100
            self._speechgen_end = t.convert_tokens_to_ids("<speechgen_end>")
        return self._speech_base, self._speechgen_end

    # Audex has no trained speaker conditioning; the voice is fully determined
    # by the sampling seed. Named voices map to seeds curated by round-tripping
    # generations through the Audex ASR model ("is the speaker male or female?").
    # All curated seeds were classified female by Audex ASR self-screening;
    # they differ in speaker timbre. Integer strings also work as raw seeds.
    VOICES: Dict[str, int] = {
        "female_1": 0,
        "female_2": 3,
        "female_3": 7,
    }
    DEFAULT_VOICE: Optional[str] = "female_2"

    def generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        cfg_scale: float = 2.0,
        max_tokens: int = 1024,
        verbose: bool = False,
        **kwargs,
    ):
        from mlx_lm.models.cache import KVCache
        from mlx_lm.sample_utils import make_sampler

        start = time.time()
        voice = voice if voice is not None else self.DEFAULT_VOICE
        if voice is not None:
            seed = self.VOICES.get(str(voice))
            if seed is None:
                try:
                    seed = int(voice)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Unknown voice {voice!r}; use one of {sorted(self.VOICES)} or an integer seed"
                    )
            mx.random.seed(seed)
        base, end_id = self._codec_token_map()
        cond_ids, null_ids = self._prompts(text)
        n_layers = len(self.model.layers)

        cond_cache = [KVCache() for _ in range(n_layers)]
        null_cache = [KVCache() for _ in range(n_layers)] if cfg_scale > 1.0 else None

        sampler = make_sampler(temperature, top_p=top_p, top_k=top_k)

        # prefill
        logits = self._lm_step(cond_ids[None], cond_cache)[:, -1:]
        if null_cache is not None:
            null_logits = self._lm_step(null_ids[None], null_cache)[:, -1:]
            logits = null_logits + cfg_scale * (logits - null_logits)

        codes: List[int] = []
        for _ in range(max_tokens):
            tok = sampler(logits[:, -1] - mx.logsumexp(logits[:, -1], keepdims=True))
            tid = int(tok.item())
            if tid == end_id or tid == self.config.eos_token_id:
                break
            if base <= tid < base + 65536:
                codes.append(tid - base)
            step = mx.array([[tid]])
            logits = self._lm_step(step, cond_cache)[:, -1:]
            if null_cache is not None:
                null_logits = self._lm_step(step, null_cache)[:, -1:]
                logits = null_logits + cfg_scale * (logits - null_logits)

        if not codes:
            raise RuntimeError("Audex TTS generated no speech codec tokens")

        wav = self.decoder(mx.array(codes)[None])
        wav = np.array(wav[0].astype(mx.float32))
        elapsed = time.time() - start
        dur = len(wav) / self.sample_rate

        yield GenerationResult(
            audio=mx.array(wav),
            samples=len(wav),
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(codes),
            audio_duration=f"{dur:.2f}s",
            real_time_factor=elapsed / dur if dur > 0 else 0.0,
            prompt={"text": text},
            audio_samples={},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            # ASR-side modules are not part of the TTS graph
            if k.startswith(("audio_encoder.", "audio_projector.")):
                continue
            # speech decoder key remapping (decoder.safetensors namespaces)
            if k.startswith("audex_speech_token_embedder."):
                out["decoder.embedder." + k.split(".", 1)[1]] = v
                continue
            if k.startswith("module."):
                r = k[len("module."):]
                r = r.replace("backbone.transformers.", "blocks.")
                r = r.replace("backbone.final_layer_norm", "final_layer_norm")
                r = r.replace(".att.", ".att.")
                r = r.replace(".mlp.fc1", ".fc1").replace(".mlp.fc2", ".fc2")
                r = r.replace("head.proj", "head")
                if r.startswith("wav_proj"):
                    continue  # training-only waveform conditioning
                if "lookahead_conv" in r or "lookahead_proj" in r:
                    # torch conv1d (out, in/groups, k) -> mlx (out, k, in/groups)
                    v = v.transpose(0, 2, 1)
                out["decoder." + r] = v
                continue
            out[k] = v
        return out

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoTokenizer

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        finally:
            transformers.logging.set_verbosity(prev)
        return model

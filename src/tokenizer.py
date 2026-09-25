"""Standalone Qwen3-TTS 12Hz codec: waveform [B,1,samples], codes [B,16,frames]."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .models.conv import (
    CausalConv1d,
    ConvNeXtBlock,
    DecoderBlock,
    DecoderCausalConv1d,
    DecoderCausalConvTranspose1d,
    SnakeBeta,
)
from .models.quantizer import (
    DecoderSplitResidualVectorQuantizer,
    MimiSplitResidualVectorQuantizer,
)
from .models.seanets import SEANetEncoder
from .models.transformer import DecoderTransformer, MimiTransformerModel


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seanet = SEANetEncoder(config)
        self.transformer = MimiTransformerModel(config)
        self.downsample = CausalConv1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=4,
            stride=2,
            bias=False,
            pad_mode="replicate",
        )
        # All 32 encoder codebooks exist in the checkpoint; the public codec
        # transmits only the first 16. Do not discard the other saved weights.
        self.quantizer = MimiSplitResidualVectorQuantizer(config)

    def forward(self, audio, num_quantizers):
        x = self.seanet(audio)
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        return self.quantizer.encode(self.downsample(x), num_quantizers).transpose(0, 1)


class AudioDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantizer = DecoderSplitResidualVectorQuantizer(config)
        self.pre_conv = DecoderCausalConv1d(config.codebook_dim, config.latent_dim, 3)
        self.transformer = DecoderTransformer(config)
        self.upsample = nn.ModuleList(
            nn.Sequential(
                DecoderCausalConvTranspose1d(
                    config.latent_dim, config.latent_dim, rate, rate
                ),
                ConvNeXtBlock(config.latent_dim),
            )
            for rate in config.upsampling_ratios
        )
        layers = [DecoderCausalConv1d(config.latent_dim, config.decoder_dim, 7)]
        for i, rate in enumerate(config.upsample_rates):
            layers.append(
                DecoderBlock(
                    config.decoder_dim // 2**i, config.decoder_dim // 2 ** (i + 1), rate
                )
            )
        out_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        layers.extend([SnakeBeta(out_dim), DecoderCausalConv1d(out_dim, 1, 7)])
        # This is the Qwen Snake convolution decoder, not a SEANetDecoder.
        self.waveform = nn.Sequential(*layers)

    def forward(self, codes):
        x = self.pre_conv(self.quantizer.decode(codes))
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        for block in self.upsample:
            x = block(x)
        return self.waveform(x).clamp(-1, 1)


class Qwen3TTSTokenizer(nn.Module):
    """Offline encoding and chunked decoding with left context.
    
    Use equal-length mono batches; encode variable-length audio separately."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.input_sample_rate = config["input_sample_rate"]
        self.output_sample_rate = config["output_sample_rate"]
        self.encode_downsample_rate = config["encode_downsample_rate"]
        self.decode_upsample_rate = config["decode_upsample_rate"]
        self.num_quantizers = config["encoder_valid_num_quantizers"]
        self.codebook_size = config["decoder_config"]["codebook_size"]
        self.encoder = AudioEncoder(SimpleNamespace(**config["encoder_config"]))
        self.decoder = AudioDecoder(SimpleNamespace(**config["decoder_config"]))

    def encode(self, audio: torch.Tensor, *, sample_rate: int | None = None):
        """[B,1,samples] -> [B,16,frames]. Resample audio before encoding."""
        if sample_rate is not None and sample_rate != self.input_sample_rate:
            raise ValueError(f"Resample input to {self.input_sample_rate} Hz first")
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        return self.encoder(audio, self.num_quantizers)

    def decode(
        self,
        codes: torch.Tensor,
        *,
        target_length: int | None = None,
        chunk_size: int | None = 300,
        left_context_size: int = 25,
    ):
        """[B,16,frames] -> [B,1,samples], with optional chunking and trimming."""
        if chunk_size is None:
            audio = self.decoder(codes)
        else:
            chunks = []
            for start in range(0, codes.shape[-1], chunk_size):
                context = min(start, left_context_size)
                chunk = self.decoder(codes[..., start - context : start + chunk_size])
                chunks.append(chunk[..., context * self.decode_upsample_rate :])
            audio = torch.cat(chunks, dim=-1)
        return audio if target_length is None else audio[..., :target_length]

    def forward(self, audio, *, sample_rate=None):
        codes = self.encode(audio, sample_rate=sample_rate)
        return codes, self.decode(codes, target_length=audio.shape[-1])

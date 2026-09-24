# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

from diffusers.loaders.lora_conversion_utils import _convert_fal_kontext_lora_to_diffusers


def _fal_kontext_state_dict(rank=1, inner_dim=3072, mlp_hidden_dim=12288, num_layers=19, num_single_layers=38):
    """Minimal fal-kontext LoRA (block keys only), shaped so the qkv / linear1 splits work."""
    prefix = "base_model.model."
    sd = {}
    for i in range(num_layers):
        for module in ["img_mod.lin", "txt_mod.lin", "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"]:
            sd[f"{prefix}double_blocks.{i}.{module}.lora_A.weight"] = torch.zeros(rank, 1)
            sd[f"{prefix}double_blocks.{i}.{module}.lora_B.weight"] = torch.zeros(1, rank)
        for module in ["img_attn.proj", "txt_attn.proj"]:
            sd[f"{prefix}double_blocks.{i}.{module}.lora_A.weight"] = torch.zeros(rank, 1)
            sd[f"{prefix}double_blocks.{i}.{module}.lora_B.weight"] = torch.zeros(1, rank)
        for module in ["img_attn.qkv", "txt_attn.qkv"]:
            sd[f"{prefix}double_blocks.{i}.{module}.lora_A.weight"] = torch.zeros(rank, 1)
            sd[f"{prefix}double_blocks.{i}.{module}.lora_B.weight"] = torch.zeros(3 * inner_dim, rank)
    for i in range(num_single_layers):
        sd[f"{prefix}single_blocks.{i}.modulation.lin.lora_A.weight"] = torch.zeros(rank, 1)
        sd[f"{prefix}single_blocks.{i}.modulation.lin.lora_B.weight"] = torch.zeros(1, rank)
        sd[f"{prefix}single_blocks.{i}.linear1.lora_A.weight"] = torch.zeros(rank, 1)
        sd[f"{prefix}single_blocks.{i}.linear1.lora_B.weight"] = torch.zeros(3 * inner_dim + mlp_hidden_dim, rank)
        sd[f"{prefix}single_blocks.{i}.linear2.lora_A.weight"] = torch.zeros(rank, 1)
        sd[f"{prefix}single_blocks.{i}.linear2.lora_B.weight"] = torch.zeros(1, rank)
    sd[f"{prefix}final_layer.linear.lora_A.weight"] = torch.zeros(rank, 1)
    sd[f"{prefix}final_layer.linear.lora_B.weight"] = torch.zeros(1, rank)
    return sd


UNPREFIXED_GLOBAL_KEYS = {
    "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
    "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
    "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
    "txt_in": "context_embedder",
    "img_in": "x_embedder",
    "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
}


def test_fal_kontext_conversion_blocks_only():
    converted = _convert_fal_kontext_lora_to_diffusers(_fal_kontext_state_dict())
    assert all(k.startswith("transformer.") for k in converted)
    assert "transformer.transformer_blocks.0.attn.to_q.lora_B.weight" in converted
    assert "transformer.single_transformer_blocks.37.proj_mlp.lora_B.weight" in converted


@pytest.mark.parametrize("lora_key", ["lora_A", "lora_B"])
def test_fal_kontext_conversion_accepts_unprefixed_global_embedders(lora_key):
    sd = _fal_kontext_state_dict()
    for src in UNPREFIXED_GLOBAL_KEYS:
        sd[f"{src}.{lora_key}.weight"] = torch.ones(1, 1)

    converted = _convert_fal_kontext_lora_to_diffusers(sd)

    for src, dst in UNPREFIXED_GLOBAL_KEYS.items():
        assert f"transformer.{dst}.{lora_key}.weight" in converted, src
        assert torch.equal(converted[f"transformer.{dst}.{lora_key}.weight"], torch.ones(1, 1))

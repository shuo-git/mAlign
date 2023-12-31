# coding=utf-8
# Copyright 2022 The OpenBMB team.
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
import torch
from .config import Config

class LlamaConfig(Config):
    """
    This is a configuration class that stores the configuration of the LLaMa model, which inherits from the Config class.
    It is used to instantiate the Bert model according to the specified parameters and define the model architecture.
    You can set specific parameters to control the output of the model.

    For example:
    [`dim_model`] is used to determine the Dimension of the encoder layers.
    You can choose to use the default value of 4096 or customize their dimensions.  
    
    """
    def __init__(self, vocab_size=32000,
                       dim_model=4096,
                       num_heads=32,
                       num_heads_kv=-1,
                       dim_head=128,
                       dim_ff=11008,
                       num_layers=32,
                       dropout_p=0,
                       emb_init_mean = 0.0,
                       emb_init_std = 0.02,
                       pos_bias_type = "rotary",
                       norm_init_var = 1.0,
                       norm_bias = False,
                       norm_eps = 1e-6,
                       att_init_mean = 0.0,
                       att_init_std = 0.02,
                       att_bias = False,
                       att_mask_value = float("-inf"),
                       ffn_init_mean = 0.0,
                       ffn_init_std = 0.02,
                       ffn_bias = False,
                       ffn_activate_fn = "gated_silu",
                       proj_init_mean = 0.0,
                       proj_init_std = 0.02,
                       proj_bias = False,
                       length_scale = False,
                       attn_scale = True,
                       half = True,
                       int8 = False,
                       tied = False,
                       cls_head = None,
                       post_layer_norm = False,
                    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv != -1 else num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head
        self.post_layer_norm = post_layer_norm
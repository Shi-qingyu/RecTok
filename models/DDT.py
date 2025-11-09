import logging
from typing import *
from math import sqrt
import torch
import torch.nn as nn

from .layers import ( 
    PatchEmbed, 
    get_2d_sincos_pos_embed, 
    DDTVisionRotaryEmbeddingFast, 
    GaussianFourierEmbedding, 
    LabelEmbedder,
    DDTFinalLayer,
    LightningDDTBlock,
)
from transport import Sampler, create_transport


logger = logging.getLogger("DeTok")


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=1,
        tokenizer_patch_size=16,
        token_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        force_one_d_seq=0,
        mlp_ratio=4.0,
        label_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed: bool = True,
        sampling_method: str = "euler",
        num_sampling_steps: int = 50,
    ):
        super().__init__()
        self.input_size = img_size // tokenizer_patch_size
        self.token_channels = token_channels
        self.out_channels = token_channels
        self.num_classes = num_classes

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
        self.force_one_d_seq = force_one_d_seq
        
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        s_channel_per_token = token_channels * self.s_patch_size * self.s_patch_size
        s_input_size = self.input_size
        s_patch_size = self.s_patch_size
        x_input_size = self.input_size
        x_patch_size = self.x_patch_size
        x_channel_per_token = token_channels * self.x_patch_size * self.x_patch_size
        self.x_embedder = PatchEmbed(
            x_input_size, 
            x_patch_size, 
            x_channel_per_token, 
            self.decoder_hidden_size, 
            bias=True
        )
        self.s_embedder = PatchEmbed(
            s_input_size, 
            s_patch_size, 
            s_channel_per_token, 
            self.encoder_hidden_size, 
            bias=True
        )
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size
        ) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, self.encoder_hidden_size, label_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder.num_patches))
            # print(f"enc_half_head_dim: {enc_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.enc_feat_rope = DDTVisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            # print(f"dec_half_head_dim: {dec_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.dec_feat_rope = DDTVisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            ) for i in range(self.num_blocks)
        ])
        
        # --------------------------------------------------------------------------
        # transport and sampling setup
        time_dist_shift_dim = self.input_size * self.input_size * self.token_channels
        time_dist_shift_base = 4096
        time_dist_shift = sqrt(time_dist_shift_dim / time_dist_shift_base)
        
        self.transport = create_transport(
            prediction="velocity",
            loss_weight="none",
            time_dist_shift=time_dist_shift,
        )
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=int(num_sampling_steps),
            timestep_shift=time_dist_shift,
        )
        
        self.initialize_weights()

        # log model info
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[DiTDDT] params: {num_trainable_params:.2f}M"
        )

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size ** 2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def net(self, x, t, y, s=None):
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = nn.functional.silu(t + y)
        if s is None:
            s = self.s_embedder(x)
            if self.use_pos_embed:
                s = s + self.pos_embed
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {self.pos_embed.shape}")
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            # broadcast t to s
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t + s)
        s = self.s_projector(s)
        x = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
        x = self.final_layer(x, s)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.net(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.token_channels], model_out[:, self.token_channels:]
        eps, rest = model_out[:, :self.token_channels], model_out[:, self.token_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, y):
        """forward pass for training."""
        loss_dict = self.transport.training_losses(self.net, x, dict(y=y))
        return loss_dict["loss"].mean()

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.net(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.token_channels]
        ag_eps = ag_model_out[:, :self.token_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None):
        """generate samples using the model."""
        device = labels.device

        # prepare noise tensor
        if self.force_one_d_seq:
            z = torch.randn(n_samples, self.force_one_d_seq, self.token_channels)
        else:
            z = torch.randn(n_samples, self.token_channels, self.input_size, self.input_size)
        z = z.to(device)

        # setup classifier-free guidance
        if cfg > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([self.num_classes] * n_samples, device=device)
            labels = torch.cat([labels, y_null], 0)
            model_kwargs = dict(y=labels, cfg_scale=cfg, cfg_interval=(0, 1.0))
            model_fn = self.forward_with_cfg
        else:
            model_kwargs = dict(y=labels)
            model_fn = self.net

        # generate samples
        samples = self.sample_fn(z, model_fn, **model_kwargs)[-1]
        if cfg > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # remove null class samples
        return samples


def DiTwDDTHead_s(**kwargs):
    logger.info(f"DiTwDDTHead_b kwargs: {kwargs}")
    return DiTwDDTHead(
        img_size=kwargs.get("img_size", 256),
        patch_size=kwargs.get("patch_size", [1, 1]),
        tokenizer_patch_size=kwargs.get("tokenizer_patch_size", 16),
        token_channels=kwargs.get("token_channels", 128),
        hidden_size=kwargs.get("hidden_size", [384, 2048]),
        depth=kwargs.get("depth", [12, 2]),
        num_heads=kwargs.get("num_heads", [6, 16]),
        mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        label_dropout_prob=kwargs.get("label_dropout_prob", 0.1),
        num_classes=kwargs.get("num_classes", 1000),
        use_qknorm=kwargs.get("use_qknorm", False),
        use_swiglu=kwargs.get("use_swiglu", True),
        use_rope=kwargs.get("use_rope", True),
        use_rmsnorm=kwargs.get("use_rmsnorm", True),
        wo_shift=kwargs.get("wo_shift", False),
        use_pos_embed=kwargs.get("use_pos_embed", True),
        sampling_method=kwargs.get("sampling_method", "euler"),
        num_sampling_steps=kwargs.get("num_sampling_steps", 50),
        force_one_d_seq=kwargs.get("force_one_d_seq", 0),
    )
    

def DiTwDDTHead_xl(**kwargs):
    logger.info(f"DiTwDDTHead_xl kwargs: {kwargs}")
    return DiTwDDTHead(
        img_size=kwargs.get("img_size", 256),
        patch_size=kwargs.get("patch_size", [1, 1]),
        tokenizer_patch_size=kwargs.get("tokenizer_patch_size", 16),
        token_channels=kwargs.get("token_channels", 768),
        hidden_size=kwargs.get("hidden_size", [1152, 2048]),
        depth=kwargs.get("depth", [28, 2]),
        num_heads=kwargs.get("num_heads", [16, 16]),
        mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        label_dropout_prob=kwargs.get("label_dropout_prob", 0.1),
        num_classes=kwargs.get("num_classes", 1000),
        use_qknorm=kwargs.get("use_qknorm", False),
        use_swiglu=kwargs.get("use_swiglu", True),
        use_rope=kwargs.get("use_rope", True),
        use_rmsnorm=kwargs.get("use_rmsnorm", True),
        wo_shift=kwargs.get("wo_shift", False),
        use_pos_embed=kwargs.get("use_pos_embed", True),
        sampling_method=kwargs.get("sampling_method", "euler"),
        num_sampling_steps=kwargs.get("num_sampling_steps", 50),
        force_one_d_seq=kwargs.get("force_one_d_seq", 0),
    )


DiTDDT_models = {
    "DiTDDT_s": DiTwDDTHead_s,
    "DiTDDT_xl": DiTwDDTHead_xl,
}
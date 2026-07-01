"""CoreModel: a decoder-only transformer with a per-position latent gate.

CoreModel reuses the tested Qwen-style primitives from ``core_outline_model``
(RoPE, GQA attention, SwiGLU MLP, RMSNorm) but weaves a :class:`GatingModule`
into the decoder stack. After each *gated* layer, a per-position router chooses
between {THINK, TOOL, RESPOND, DONE} and injects the chosen mode's learned
embedding into the residual stream.

Routing is latent (no mode labels). The training loss is::

    loss = L_lm + router_aux_loss_coef * L_balance + router_z_loss_coef * L_z

Naming note: the plain ``CoreModel`` symbol already exists in
``src/models/model.py``. To avoid shadowing existing imports, the gated
architecture is exported as ``GatedCoreModel`` (backbone),
``CoreModelForCausalLM`` (LM head) and ``CoreModelConfig``.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.core.gating import GatingModule, NUM_MODES, Mode
from src.models.core_outline_model import (
    CoreOutlineConfig,
    CoreOutlineDecoderLayer,
    CoreOutlineRotaryEmbedding,
)


class CoreModelConfig(CoreOutlineConfig):
    """Configuration for the gated CoreModel.

    Extends :class:`CoreOutlineConfig` with gating hyperparameters.
    """

    def __init__(
        self,
        num_modes: int = NUM_MODES,
        gate_layer_indices: Optional[List[int]] = None,
        share_gate: bool = True,
        router_aux_loss_coef: float = 1e-2,
        router_z_loss_coef: float = 1e-3,
        gate_noise_std: float = 0.3,
        gate_temperature: float = 1.0,
        use_straight_through: bool = False,
        mode_balance_target: Optional[List[float]] = None,
        done_mode_id: int = int(Mode.DONE),
        done_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_modes = num_modes
        # Default: a single gate at the middle of the stack.
        if gate_layer_indices is None:
            gate_layer_indices = [self.num_hidden_layers // 2]
        self.gate_layer_indices = list(gate_layer_indices)
        self.share_gate = share_gate
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.gate_noise_std = gate_noise_std
        self.gate_temperature = gate_temperature
        self.use_straight_through = use_straight_through
        self.mode_balance_target = mode_balance_target
        self.done_mode_id = done_mode_id
        self.done_threshold = done_threshold


class GatedCoreModel(nn.Module):
    """Decoder backbone with gating woven into the layer stack."""

    def __init__(self, config: CoreModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gate_layer_indices = set(config.gate_layer_indices)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CoreOutlineDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = CoreOutlineRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0),
        )

        # Build the gate(s). Either one shared gate reused at every gated layer,
        # or one independent gate per gated layer.
        balance_target = (
            torch.tensor(config.mode_balance_target, dtype=torch.float)
            if config.mode_balance_target is not None
            else None
        )

        def make_gate() -> GatingModule:
            return GatingModule(
                hidden_size=config.hidden_size,
                num_modes=config.num_modes,
                noise_std=config.gate_noise_std,
                temperature=config.gate_temperature,
                use_straight_through=config.use_straight_through,
                balance_target=balance_target,
                done_mode_id=config.done_mode_id,
                expected_seq_len=min(config.max_position_embeddings, 512),
            )

        if config.share_gate:
            shared = make_gate()
            self.gates = nn.ModuleDict({str(i): shared for i in sorted(self.gate_layer_indices)})
        else:
            self.gates = nn.ModuleDict(
                {str(i): make_gate() for i in sorted(self.gate_layer_indices)}
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # --- mask helpers (identical semantics to CoreOutlineModel) ---
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_len):
        combined = None
        if input_shape[-1] > 1:
            combined = self._make_causal_mask(input_shape, inputs_embeds.dtype, inputs_embeds.device, past_len)
        if attention_mask is not None:
            expanded = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined = expanded if combined is None else expanded + combined
        return combined

    def _make_causal_mask(self, input_ids_shape, dtype, device, past_key_values_length=0):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
            )
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def post_init(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_gates: bool = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # `gate_mask` marks real (1) vs pad (0) tokens for the aux-loss stats.
        if attention_mask is None:
            gate_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)
        else:
            gate_mask = attention_mask.to(torch.bool)

        attn_mask = self._prepare_decoder_attention_mask(
            gate_mask.to(inputs_embeds.dtype), (batch_size, seq_length), inputs_embeds, 0
        )

        hidden_states = inputs_embeds
        gate_weights_all = []
        balance_total = hidden_states.new_zeros(())
        z_total = hidden_states.new_zeros(())

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

            if idx in self.gate_layer_indices:
                gate = self.gates[str(idx)]
                mode_ctx, gate_weights, aux = gate(hidden_states, attention_mask=gate_mask)
                hidden_states = hidden_states + mode_ctx
                balance_total = balance_total + aux["balance"]
                z_total = z_total + aux["z"]
                if return_gates:
                    gate_weights_all.append(gate_weights)

        hidden_states = self.norm(hidden_states)

        aux_loss = (
            self.config.router_aux_loss_coef * balance_total
            + self.config.router_z_loss_coef * z_total
        )

        gate_weights_stacked = (
            torch.stack(gate_weights_all, dim=0) if (return_gates and gate_weights_all) else None
        )

        return {
            "last_hidden_state": hidden_states,
            "gate_weights": gate_weights_stacked,  # [num_gate_layers, B, T, num_modes]
            "aux_loss": aux_loss,
            "balance_loss": balance_total.detach(),
            "z_loss": z_total.detach(),
        }


class CoreModelForCausalLM(nn.Module):
    """CoreModel with a language-modeling head and the gating aux losses."""

    def __init__(self, config: CoreModelConfig):
        super().__init__()
        self.config = config
        self.model = GatedCoreModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def post_init(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_gates: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_gates=return_gates,
        )
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        aux_loss = outputs["aux_loss"]
        loss = None
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )
            loss = lm_loss + aux_loss

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "balance_loss": outputs["balance_loss"],
            "z_loss": outputs["z_loss"],
            "logits": logits,
            "gate_weights": outputs["gate_weights"],
            "last_hidden_state": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        stop_on_done: bool = True,
    ):
        """Greedy/sampled generation that also stops on the DONE gate.

        Generation halts when any of: the DONE mode probability at the last
        position (top gate layer) exceeds ``config.done_threshold``, an EOS token
        is produced, or ``max_new_tokens`` is reached.
        """
        self.eval()
        done_id = self.config.done_mode_id
        threshold = self.config.done_threshold
        ctx = self.config.max_position_embeddings

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -ctx:]
            out = self.forward(input_ids=idx_cond, return_gates=True)
            logits = out["logits"][:, -1, :]

            if temperature and temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if stop_on_done and out["gate_weights"] is not None:
                # Top gate layer, last position, DONE probability.
                done_prob = out["gate_weights"][-1, :, -1, done_id]
                if bool((done_prob > threshold).all()):
                    break
            if eos_id is not None and bool((next_token == eos_id).all()):
                break

        return input_ids


def create_core_model(config: Optional[CoreModelConfig] = None) -> CoreModelForCausalLM:
    """Create a gated CoreModel with default or custom configuration."""
    if config is None:
        config = CoreModelConfig()
    return CoreModelForCausalLM(config)

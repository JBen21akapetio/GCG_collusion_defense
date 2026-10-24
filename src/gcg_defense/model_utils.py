"""Model loading, the move distribution q(.|prompt), and the GCG forward paths.

Three forward paths share one tokenisation of each prompt
(`prefix_ids | suffix_ids | postfix_ids`):

* `q_batch`        -- no-grad scoring of many prompts with one suffix;
* `q_batch_onehot` -- differentiable in a one-hot suffix (the GCG gradient);
* `q_candidates`   -- no-grad scoring of many suffixes on ONE prompt, reusing
                      the prefix KV cache (the GCG candidate evaluation).

B answers with a single move word, so q is read from the next-token logits
right after the assistant header, restricted to the three move tokens and
renormalised. `validity` is the probability mass those three tokens hold
before renormalisation.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .game import MOVES
from .prompts import RenderedPrompt

log = logging.getLogger(__name__)

_PRINTABLE = set(string.printable) - set("\t\r\x0b\x0c")


@dataclass
class MoveTokens:
    """How the three move words map onto single tokens."""

    ids: List[int]
    variant: str
    pieces: List[str]
    note: str

    def report(self) -> str:
        return (
            f"move tokens: variant={self.variant} ids={self.ids} "
            f"pieces={self.pieces} ({self.note})"
        )


@dataclass
class TokenizedPrompt:
    prefix_ids: List[int]
    postfix_ids: List[int]
    uid: str = ""

    def length(self, suffix_len: int) -> int:
        return len(self.prefix_ids) + suffix_len + len(self.postfix_ids)


def resolve_move_tokens(tokenizer, moves: Sequence[str] = MOVES) -> MoveTokens:
    """Pick a single-token representation of Rock / Paper / Scissors.

    Tries the bare word (what follows the assistant header), then the word with
    a leading space, then falls back to the first token of each bare word and
    requires the three first tokens to be distinct.
    """
    for variant, fmt in (("plain", "{}"), ("leading_space", " {}")):
        ids: List[int] = []
        for move in moves:
            enc = tokenizer.encode(fmt.format(move), add_special_tokens=False)
            if len(enc) != 1:
                ids = []
                break
            ids.append(int(enc[0]))
        if len(ids) == len(moves) and len(set(ids)) == len(moves):
            return MoveTokens(
                ids=ids,
                variant=variant,
                pieces=[tokenizer.decode([i]) for i in ids],
                note="each move word is a single token",
            )
    ids = [
        int(tokenizer.encode(move, add_special_tokens=False)[0]) for move in moves
    ]
    if len(set(ids)) != len(moves):
        raise RuntimeError(
            f"cannot separate the move words with first tokens: {ids}"
        )
    pieces = [tokenizer.decode([i]) for i in ids]
    return MoveTokens(
        ids=ids,
        variant="first_token",
        pieces=pieces,
        note="not all move words are single tokens; using distinct first tokens",
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        if dtype != "float32":
            log.warning("dtype %s requested on CPU; falling back to float32", dtype)
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


class MoveModel:
    """A frozen causal LM exposing B's move distribution."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        moves: Sequence[str] = MOVES,
        batch_size: int = 64,
        use_kv_cache: bool = True,
        max_batch_size: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype, self.device)
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.candidate_batch_size = batch_size
        self.use_kv_cache = use_kv_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.moves = list(moves)
        self.move_tokens = resolve_move_tokens(self.tokenizer, self.moves)
        self.move_ids = torch.tensor(self.move_tokens.ids, device=self.device)
        self.core = getattr(self.model, self.model.base_model_prefix)
        self.lm_head = self.model.get_output_embeddings()
        self.embedding = self.model.get_input_embeddings()
        self.vocab_size = int(self.embedding.weight.shape[0])
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        self.pad_id = int(pad_id)
        self._allowed_mask: Optional[torch.Tensor] = None

    # -- tokenisation ------------------------------------------------------
    def tokenize(self, rendered: RenderedPrompt, uid: str = "") -> TokenizedPrompt:
        prefix = self.tokenizer.encode(rendered.prefix, add_special_tokens=False)
        postfix = (
            self.tokenizer.encode(rendered.postfix, add_special_tokens=False)
            if rendered.postfix
            else []
        )
        return TokenizedPrompt(list(prefix), list(postfix), uid)

    def encode_suffix(self, suffix: str) -> List[int]:
        return list(self.tokenizer.encode(suffix, add_special_tokens=False))

    def decode_suffix(self, suffix_ids: Sequence[int]) -> str:
        # clean_up_tokenization_spaces must stay off: it deletes the space
        # before punctuation, which breaks the decode -> re-encode round trip
        # for almost every candidate (measured survival 0% with it, 82%
        # without) and would silently starve the GCG search.
        return self.tokenizer.decode(
            list(suffix_ids), clean_up_tokenization_spaces=False
        )

    def suffix_roundtrips(self, suffix_ids: Sequence[int]) -> bool:
        """True iff decode -> re-encode gives back exactly the same token ids."""
        text = self.decode_suffix(suffix_ids)
        return self.encode_suffix(text) == list(suffix_ids)

    def allowed_token_mask(self, allow_non_ascii: bool = False) -> torch.Tensor:
        """[V] bool mask of tokens GCG may place in the suffix."""
        if self._allowed_mask is not None:
            return self._allowed_mask
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        special = set(self.tokenizer.all_special_ids)
        for extra in getattr(self.tokenizer, "added_tokens_decoder", {}):
            special.add(int(extra))
        for tid in special:
            if 0 <= tid < self.vocab_size:
                mask[tid] = False
        pieces = self.tokenizer.batch_decode(
            [[i] for i in range(self.vocab_size)], skip_special_tokens=False
        )
        for tid, piece in enumerate(pieces):
            if not piece or piece.strip() == "":
                mask[tid] = False
            elif not allow_non_ascii and any(ch not in _PRINTABLE for ch in piece):
                mask[tid] = False
        self._allowed_mask = mask.to(self.device)
        return self._allowed_mask

    # -- low level forward -------------------------------------------------
    def _last_hidden(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        want_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], object]:
        """Run the base transformer and return (last hidden, layer hidden, cache).

        Calling the base model (not the LM head) keeps the [B, T, V] logits
        tensor from ever being materialised.
        """
        out = self.core(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            output_hidden_states=want_layer is not None,
        )
        hidden = out.last_hidden_state[:, -1, :]
        layer_hidden = None
        if want_layer is not None:
            layer_hidden = out.hidden_states[want_layer][:, -1, :]
        return hidden, layer_hidden, getattr(out, "past_key_values", None)

    def _q_from_hidden(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move distribution and validity from the final hidden state."""
        logits = self.lm_head(hidden).float()
        log_probs = F.log_softmax(logits, dim=-1)
        move_log_probs = log_probs.index_select(-1, self.move_ids)
        validity = move_log_probs.logsumexp(dim=-1).exp()
        q = F.softmax(move_log_probs, dim=-1)
        return q, validity

    # -- path 1: many prompts, one suffix ----------------------------------
    def _pack(
        self, prompts: Sequence[TokenizedPrompt], suffix_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-padded input_ids / attention_mask / position_ids."""
        rows = [list(p.prefix_ids) + list(suffix_ids) + list(p.postfix_ids) for p in prompts]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, width - len(row) :] = torch.tensor(row, dtype=torch.long)
            mask[i, width - len(row) :] = 1
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    @torch.no_grad()
    def q_batch(
        self,
        prompts: Sequence[TokenizedPrompt],
        suffix_ids: Sequence[int] = (),
        want_layer: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """q [n, 3], validity [n] and optionally the residual stream [n, d]."""
        bs = batch_size or self.batch_size
        qs, vals, acts, hiddens = [], [], [], []
        for start in range(0, len(prompts), bs):
            chunk = prompts[start : start + bs]
            ids, mask, pos = self._pack(chunk, suffix_ids)
            hidden, layer_hidden, _ = self._last_hidden(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                want_layer=want_layer,
            )
            q, validity = self._q_from_hidden(hidden)
            qs.append(q)
            vals.append(validity)
            if layer_hidden is not None:
                acts.append(layer_hidden.float())
            if return_hidden:
                hiddens.append(hidden)
        out = {"q": torch.cat(qs), "validity": torch.cat(vals)}
        if acts:
            out["acts"] = torch.cat(acts)
        if hiddens:
            out["hidden"] = torch.cat(hiddens)
        return out

    # -- path 2: differentiable in the suffix ------------------------------
    def q_batch_onehot(
        self,
        prompts: Sequence[TokenizedPrompt],
        one_hot: torch.Tensor,
        want_layer: Optional[int] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Same as `q_batch` but differentiable w.r.t. a [k, V] one-hot suffix."""
        suffix_embeds = one_hot @ self.embedding.weight
        embeds, mask, pos = self._pack_embeds(prompts, suffix_embeds)
        hidden, layer_hidden, _ = self._last_hidden(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=pos,
            want_layer=want_layer,
        )
        q, validity = self._q_from_hidden(hidden)
        out = {"q": q, "validity": validity}
        if layer_hidden is not None:
            out["acts"] = layer_hidden.float()
        if return_hidden:
            out["hidden"] = hidden
        return out

    def _pack_embeds(
        self, prompts: Sequence[TokenizedPrompt], suffix_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = suffix_embeds.shape[0]
        weight = self.embedding.weight
        pad_embed = weight[self.pad_id]
        lengths = [p.length(k) for p in prompts]
        width = max(lengths)
        rows, masks = [], []
        for prompt, length in zip(prompts, lengths):
            pieces = []
            if width > length:
                pieces.append(pad_embed.unsqueeze(0).expand(width - length, -1))
            if prompt.prefix_ids:
                pieces.append(
                    weight[torch.tensor(prompt.prefix_ids, device=self.device)]
                )
            pieces.append(suffix_embeds.to(weight.dtype))
            if prompt.postfix_ids:
                pieces.append(
                    weight[torch.tensor(prompt.postfix_ids, device=self.device)]
                )
            rows.append(torch.cat(pieces, dim=0))
            masks.append([0] * (width - length) + [1] * length)
        embeds = torch.stack(rows, dim=0)
        mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return embeds, mask, pos

    # -- GPU saturation ----------------------------------------------------
    def gpu_memory_report(self) -> str:
        if self.device.type != "cuda":
            return "cpu"
        free, total = torch.cuda.mem_get_info(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        return (
            f"peak_alloc={peak / 2**30:.2f}GiB "
            f"used={(total - free) / 2**30:.2f}/{total / 2**30:.1f}GiB"
        )

    def autotune_candidate_batch(
        self, prompt: "TokenizedPrompt", suffix_len: int, cap: Optional[int] = None
    ) -> int:
        """Largest candidate batch that fits, found by doubling until OOM.

        Candidate scoring is the dominant cost of a GCG step, so running it at
        the largest batch the card will take is what keeps the GPU saturated.
        """
        if self.device.type != "cuda":
            self.candidate_batch_size = min(self.batch_size, cap or self.batch_size)
            return self.candidate_batch_size
        cap = min(cap or self.max_batch_size, self.max_batch_size)
        probe = torch.zeros((1, suffix_len), dtype=torch.long, device=self.device)
        best = 1
        size = max(1, self.batch_size // 4)
        while size <= cap:
            candidates = probe.expand(size, -1).contiguous()
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                self.q_candidates(prompt, candidates, batch_size=size)
                best = size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            except RuntimeError as exc:  # pragma: no cover - driver-specific
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                break
            size *= 2
        self.candidate_batch_size = best
        log.info(
            "candidate batch autotuned to %d (cap %d) | %s",
            best,
            cap,
            self.gpu_memory_report(),
        )
        torch.cuda.empty_cache()
        return best

    # -- path 3: many suffixes, one prompt (prefix KV cache) ---------------
    @torch.no_grad()
    def q_candidates(
        self,
        prompt: TokenizedPrompt,
        candidates: torch.Tensor,
        want_layer: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """q [C, 3] and validity [C] for C candidate suffixes on one prompt."""
        bs = batch_size or self.candidate_batch_size
        n_cand, k = candidates.shape
        prefix = torch.tensor([prompt.prefix_ids], dtype=torch.long, device=self.device)
        postfix = torch.tensor(prompt.postfix_ids, dtype=torch.long, device=self.device)
        prefix_len = prefix.shape[1]
        cache = None
        if self.use_kv_cache and prefix_len > 0:
            mask = torch.ones_like(prefix)
            pos = torch.arange(prefix_len, device=self.device).unsqueeze(0)
            try:
                _, _, cache = self._last_hidden(
                    input_ids=prefix,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_values=_empty_cache(),
                )
            except Exception as exc:  # pragma: no cover - version guard
                log.warning("prefix KV cache unavailable (%s); using full forwards", exc)
                self.use_kv_cache = False
                cache = None
        qs, vals, acts, hiddens = [], [], [], []
        for start in range(0, n_cand, bs):
            chunk = candidates[start : start + bs].to(self.device)
            c = chunk.shape[0]
            tail = postfix.unsqueeze(0).expand(c, -1)
            tokens = torch.cat([chunk, tail], dim=1)
            if cache is not None:
                expanded = _expand_cache(cache, c)
                if expanded is None:
                    self.use_kv_cache = False
                    cache = None
                else:
                    mask = torch.ones(
                        (c, prefix_len + tokens.shape[1]),
                        dtype=torch.long,
                        device=self.device,
                    )
                    pos = (
                        torch.arange(
                            prefix_len,
                            prefix_len + tokens.shape[1],
                            device=self.device,
                        )
                        .unsqueeze(0)
                        .expand(c, -1)
                    )
                    hidden, layer_hidden, _ = self._last_hidden(
                        input_ids=tokens,
                        attention_mask=mask,
                        position_ids=pos,
                        past_key_values=expanded,
                        want_layer=want_layer,
                    )
                    q, validity = self._q_from_hidden(hidden)
                    qs.append(q)
                    vals.append(validity)
                    if layer_hidden is not None:
                        acts.append(layer_hidden.float())
                    if return_hidden:
                        hiddens.append(hidden)
                    continue
            full = torch.cat([prefix.expand(c, -1), tokens], dim=1)
            mask = torch.ones_like(full)
            pos = torch.arange(full.shape[1], device=self.device).unsqueeze(0).expand(c, -1)
            hidden, layer_hidden, _ = self._last_hidden(
                input_ids=full,
                attention_mask=mask,
                position_ids=pos,
                want_layer=want_layer,
            )
            q, validity = self._q_from_hidden(hidden)
            qs.append(q)
            vals.append(validity)
            if layer_hidden is not None:
                acts.append(layer_hidden.float())
            if return_hidden:
                hiddens.append(hidden)
        out = {"q": torch.cat(qs), "validity": torch.cat(vals)}
        if acts:
            out["acts"] = torch.cat(acts)
        if hiddens:
            out["hidden"] = torch.cat(hiddens)
        return out

    @torch.no_grad()
    def all_layer_acts(
        self,
        prompts: Sequence[TokenizedPrompt],
        suffix_ids: Sequence[int] = (),
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """[n, n_layers + 1, d] last-token residual stream at every layer.

        One forward pass per batch instead of one per layer, which is what the
        diff-in-means direction scan needs.
        """
        bs = batch_size or self.batch_size
        chunks = []
        for start in range(0, len(prompts), bs):
            ids, mask, pos = self._pack(prompts[start : start + bs], suffix_ids)
            out = self.core(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                use_cache=False,
                output_hidden_states=True,
            )
            stacked = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
            chunks.append(stacked.float())
        return torch.cat(chunks)

    # -- misc --------------------------------------------------------------
    @torch.no_grad()
    def token_logprob(
        self,
        prompts: Sequence[TokenizedPrompt],
        suffix_ids: Sequence[int],
        token_id: int,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """log p(token_id) at the answer position (used by the positive control)."""
        bs = batch_size or self.batch_size
        outs = []
        for start in range(0, len(prompts), bs):
            ids, mask, pos = self._pack(prompts[start : start + bs], suffix_ids)
            hidden, _, _ = self._last_hidden(
                input_ids=ids, attention_mask=mask, position_ids=pos
            )
            logits = self.lm_head(hidden).float()
            outs.append(F.log_softmax(logits, dim=-1)[:, token_id])
        return torch.cat(outs)

    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def _empty_cache():
    """A fresh Cache object, or None on versions that take legacy tuples."""
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache()
    except Exception:  # pragma: no cover - very old transformers
        return None


def _expand_cache(cache, batch_size: int):
    """Expand a batch-1 KV cache along the batch dimension.

    transformers has moved the Cache internals several times, so try the
    documented helpers first and fall back to the legacy tuple round-trip.
    `None` means "give up and recompute the prefix".
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:  # pragma: no cover
        return None
    try:
        legacy = _cache_to_legacy(cache)
        if legacy is None:
            return None
        expanded = tuple(
            (
                k.expand(batch_size, -1, -1, -1).contiguous(),
                v.expand(batch_size, -1, -1, -1).contiguous(),
            )
            for k, v in legacy
        )
        return DynamicCache.from_legacy_cache(expanded)
    except Exception as exc:  # pragma: no cover - version guard
        log.warning("could not expand the KV cache (%s)", exc)
        return None


def _cache_to_legacy(cache):
    if isinstance(cache, tuple):
        return cache
    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        return tuple(to_legacy())
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is not None and values is not None:
        return tuple(zip(keys, values))
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return tuple((layer.keys, layer.values) for layer in layers)
    return None

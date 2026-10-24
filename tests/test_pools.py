"""Prompt pools: ordering of A's moves, subsetting and the no-suffix twins."""

import numpy as np
import pytest

from gcg_defense import prompts
from gcg_defense.pools import build_pool


class FakeTokenizer:
    """Just enough chat template and encoding to exercise the pool logic."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return body + "<|assistant|>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 for c in text]


class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def tokenize(self, rendered, uid=""):
        from gcg_defense.model_utils import TokenizedPrompt

        return TokenizedPrompt(
            self.tokenizer.encode(rendered.prefix),
            self.tokenizer.encode(rendered.postfix) if rendered.postfix else [],
            uid,
        )


@pytest.fixture
def model():
    return FakeModel()


def _contexts(condition, n=4):
    return prompts.build_contexts(condition, n, 1, np.random.default_rng(0))


def test_colluding_pool_has_three_prompts_per_context_in_move_order(model):
    contexts = _contexts("colluding")
    pool = build_pool(model, "colluding", contexts, "system_end")
    assert pool.per_context == 3
    assert len(pool.prompts) == 3 * len(contexts)
    assert [p.uid.split("a=")[1] for p in pool.prompts[:3]] == ["0", "1", "2"]


def test_independent_pool_has_one_prompt_per_context(model):
    pool = build_pool(model, "independent", _contexts("independent"), "system_end")
    assert pool.per_context == 1
    assert len(pool.prompts) == pool.n_contexts


def test_base_prompts_have_no_suffix_slot(model):
    pool = build_pool(model, "colluding", _contexts("colluding"), "system_end")
    for prompt, base in zip(pool.prompts, pool.base_prompts):
        assert base.postfix_ids == []          # nothing after the slot
        assert prompt.postfix_ids != []        # the slot splits the text
        assert len(base.prefix_ids) < prompt.length(0) + 1


def test_subset_keeps_prompts_aligned_with_contexts(model):
    pool = build_pool(model, "colluding", _contexts("colluding", 6), "system_end")
    sub = pool.subset([1, 3])
    assert sub.n_contexts == 2
    assert [c.uid for c in sub.contexts] == [pool.contexts[1].uid, pool.contexts[3].uid]
    assert sub.prompts == pool.prompts[3:6] + pool.prompts[9:12]


def test_reshape_groups_by_context():
    import torch

    from gcg_defense.pools import PromptPool

    pool = PromptPool("x", [1, 2], [None] * 6, [None] * 6, 3)
    flat = torch.arange(18).reshape(6, 3)
    assert pool.reshape(flat).shape == (2, 3, 3)
    assert torch.equal(pool.reshape(flat)[1, 0], flat[3])


def test_user_end_places_the_slot_in_the_user_turn(model):
    pool = build_pool(model, "colluding", _contexts("colluding"), "user_end")
    system_end = build_pool(model, "colluding", _contexts("colluding"), "system_end")
    # with the slot at the very end of the user turn, only the assistant header
    # follows it, so the postfix is much shorter than for system_end
    assert len(pool.prompts[0].postfix_ids) < len(system_end.prompts[0].postfix_ids)


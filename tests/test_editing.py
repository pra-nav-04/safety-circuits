"""build_suppression_examples: supervise only the affirmative-target tokens.

CPU/torch-only with a fake tokenizer — no model downloads, no TransformerLens.
"""

from types import SimpleNamespace

from safety_circuits.editing import _as_id_list, build_suppression_examples


class FakeTokenizer:
    chat_template = "x"          # truthy → use the chat-template branch
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False,
                            return_dict=False, enable_thinking=None):
        assert add_generation_prompt and tokenize
        return [10, 11, 12]      # canonical prompt ids (bare list)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [20, 21]}


class DictTokenizer(FakeTokenizer):
    # mirrors transformers versions where apply_chat_template returns a BatchEncoding-like
    # dict instead of a bare list — the case that caused "too many dimensions 'str'".
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False,
                            return_dict=False, enable_thinking=None):
        return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}


def test_as_id_list_normalises_shapes():
    assert _as_id_list([1, 2, 3]) == [1, 2, 3]
    assert _as_id_list({"input_ids": [1, 2]}) == [1, 2]
    assert _as_id_list([[1, 2, 3]]) == [1, 2, 3]          # leading batch dim


def test_dict_returning_template_does_not_leak_str_keys():
    spec = SimpleNamespace(no_think=False)
    ex = build_suppression_examples(DictTokenizer(), spec, [("q", "t")], max_target_tokens=8)
    # input_ids must be all ints (no 'input_ids'/'attention_mask' string keys)
    assert all(isinstance(t, int) for t in ex[0]["input_ids"])
    assert ex[0]["input_ids"] == [10, 11, 12, 20, 21, 99]


def test_label_mask_supervises_only_target():
    spec = SimpleNamespace(no_think=False)
    ex = build_suppression_examples(FakeTokenizer(), spec, [("harmful?", "Sure, here")], max_target_tokens=8)
    assert len(ex) == 1
    e = ex[0]
    assert e["input_ids"] == [10, 11, 12, 20, 21, 99]    # prompt + target + eos
    assert e["labels"] == [-100, -100, -100, 20, 21, 99]  # prompt masked, target supervised


def test_target_truncation():
    spec = SimpleNamespace(no_think=False)
    ex = build_suppression_examples(FakeTokenizer(), spec, [("q", "t")], max_target_tokens=1)
    # target truncated to 1 token, then eos appended
    assert ex[0]["input_ids"] == [10, 11, 12, 20, 99]
    assert ex[0]["labels"][:3] == [-100, -100, -100]

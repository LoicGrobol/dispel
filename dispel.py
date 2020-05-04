from __future__ import annotations

import functools
from typing import Callable, List, NamedTuple, Sequence, Tuple, Union

import torch
import transformers


class EncodedBatch(NamedTuple):
    """A retokenized and digitized batch of sentences.

    Attributes
    ==========

    :encoded: the batch as encoded by `transformers.batch_encode_plus`, ready to be fed to a
    transformer model  
    :alignments: the indices of the subwords corresponding to the j-th word of the i-th sentence in
    the batch will be at `subword_indices[i, alignments[i][j][0]:alignments[i][j][1], ...]`
    """

    batch: transformers.tokenization_utils.BatchEncoding
    alignments: Sequence[Sequence[Tuple[int, int]]]


class VectorizedSentence(NamedTuple):
    """A Sentence as vectorized by a transformer. The fields mirror those of the transformer output.
    
    Attributes
    ==========
    
    :last_hidden_state: `torch.Tensor` of shape `(len(sentence), embeddings_dim)`  
    :pooler_output: `torch.Tensor` of shape `(embeddings_dim,)`  
    :hidden_states: tuple of length `n_layers` of `torch.Tensor`s of shape 
      `(len(sentence), embeddings_dim)`
    """

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]


# **1** specifying special added tokens and **0** specifying sequence tokens.
# [the
# doc](https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.batch_encode_plus)
# says the opposite but that's what the code does
def align_with_special_tokens(
    word_lengths: Sequence[int],
    mask=Sequence[int],
    special_tokens_code: int = 1,
    sequence_tokens_code: int = 0,
) -> List[Tuple[int, int]]:
    res: List[Tuple[int, int]] = []
    pos = 0
    for length in word_lengths:
        while mask[pos] == special_tokens_code:
            pos += 1
        word_end = pos + length
        if any(token_type != sequence_tokens_code for token_type in mask[pos:word_end]):
            raise ValueError(
                "mask incompatible with tokenization:"
                f" needed {length} true tokens (1) at position {pos},"
                f" got {mask[pos:word_end]} instead"
            )
        res.append((pos, word_end))
        pos = word_end

    return res


class Vectorizer(torch.nn.Module):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        reduction: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
            torch.sum, dim=0
        ),
    ):
        super().__init__()
        self.model = model
        self.reduction = reduction
        self.tokenizer = tokenizer

    def encode(self, sentences: Sequence[Sequence[str]]) -> EncodedBatch:
        batch_subwords: List[List[str]] = []
        word_lengths: List[List[int]] = []
        for sent in sentences:
            sent_subwords: List[str] = []
            sent_lengths: List[int] = []
            for word in sent:
                subwords = self.tokenizer.tokenize(word)
                sent_subwords.extend(subwords)
                sent_lengths.append(len(subwords))
            batch_subwords.append(sent_subwords)
            word_lengths.append(sent_lengths)
        encoded = self.tokenizer.batch_encode_plus(
            batch_subwords,
            add_special_tokens=True,
            is_pretokenized=True,
            pad_to_max_length=True,
            return_special_tokens_masks=True,
            return_tensors="pt",
        )
        alignments = [
            align_with_special_tokens(l, m)
            for (l, m) in zip(word_lengths, encoded["special_tokens_mask"])
        ]
        return EncodedBatch(batch=encoded, alignments=alignments)

    def forward(self, sentences: Sequence[Sequence[str]]) -> List[VectorizedSentence]:
        encoded = self.encode(sentences)
        pooler_output: torch.Tensor  # Transformers output typing is a bit wild
        last_hidden_state, pooler_output, hidden_states = self.model(
            input_ids=encoded.batch["input_ids"],
            attention_mask=encoded.batch["attention_mask"],
            token_type_ids=encoded.batch["token_type_ids"],
        )
        batch_out = []
        for last_layer, *all_layers, sent_align in zip(
            last_hidden_state, *hidden_states, encoded.alignments
        ):
            sent_reduced_last = []
            sent_reduced_all = []
            for word_start, word_end in sent_align:
                reduced_last = self.reduction(last_layer[word_start:word_end])
                sent_reduced_last.append(reduced_last)
                reduced_all = [
                    self.reduction(l[word_start:word_end]) for l in all_layers
                ]
                sent_reduced_all.append(reduced_all)
            batch_out.append(
                VectorizedSentence(
                    last_hidden_state=torch.stack(sent_reduced_last, dim=0),
                    pooler_output=pooler_output,
                    hidden_states=tuple(
                        torch.stack(l, dim=0) for l in zip(*sent_reduced_all)
                    ),
                )
            )
        return batch_out

    def freeze(self, freezing: bool = True):
        def no_train(model, mode=True):
            return model

        if freezing:
            self.model.eval()
            self.model.train = no_train
            for p in self.model.network.parameters():
                p.requires_grad = False
        else:
            for p in self.model.network.parameters():
                p.requires_grad = False
            self.model.train = type(self.model).train
            self.model.train()

    def unfreeze(self):
        self.freeze(freezing=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        reduction: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
            torch.sum, dim=0
        ),
    ) -> Vectorizer:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, output_hidden_states=True
        )
        model = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=config
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        return cls(model=model, tokenizer=tokenizer, reduction=reduction)


if __name__ == "__main__":
    v = Vectorizer.from_pretrained("camembert-base")
    sents = [
        ["Je", "reconnais", "l'", "existence", "du", "kiwi"],
        ["Moi", "ma", "mère", "la", "vaisselle", "c'", "est", "Ikea"],
        ["Je", "suis", "con-", "euh", "concentré"],
        [
            "se",
            "le",
            "virent",
            "si",
            "bele",
            "qu'",
            "il",
            "en",
            "furent",
            "tot",
            "esmari",
        ],
    ]
    v.eval()
    assert all(not p.requires_grad for p in v.parameters())
    b = v(sents)
    print(b)
    assert len(b) == len(sents)
    for so, se in zip(sents, b):
        assert se.last_hidden_state.shape[:-1] == torch.Size([len(so)])
        assert all(l.shape[:-1] == torch.Size([len(so)]) for l in se.hidden_states)

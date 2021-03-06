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


# TODO: it should be possible to jit this, but how to deal with the reduction?
def reduce_chunks(
    sequence: torch.Tensor,
    alignment: List[Tuple[int, int]],
    reduction: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    res_lst: List[torch.Tensor] = []
    for start, end in alignment:
        res_lst.append(reduction(sequence[start:end, ...]))
    return torch.stack(res_lst, dim=0)


class Vectorizer(torch.nn.Module):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        reduction: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
            torch.mean, dim=0
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

    def forward(self, inpt: EncodedBatch) -> List[VectorizedSentence]:
        pooler_output: torch.Tensor  # Transformers output typing is a bit wild
        last_hidden_state, pooler_output, hidden_states = self.model(
            input_ids=inpt.batch["input_ids"],
            attention_mask=inpt.batch["attention_mask"],
        )
        batch_out = []
        for last_layer, *all_layers, sent_align in zip(
            last_hidden_state, *hidden_states, inpt.alignments
        ):
            sent_reduced_last = reduce_chunks(last_layer, sent_align, self.reduction)
            sent_reduced_all = tuple(
                reduce_chunks(l, sent_align, self.reduction) for l in all_layers
            )
            batch_out.append(
                VectorizedSentence(
                    last_hidden_state=sent_reduced_last,
                    pooler_output=pooler_output,
                    hidden_states=tuple(
                        torch.stack(l, dim=0) for l in zip(*sent_reduced_all)
                    ),
                )
            )
        return batch_out

    def vectorize(self, sentences: Sequence[Sequence[str]]) -> List[VectorizedSentence]:
        """Convenience function that encodes and vectorizes a batch of tokenized sentences."""
        encoded = self.encode(sentences)
        return self(encoded)

    def freeze(self, freezing: bool = True):
        """Make the underlying transformer model either finutunable or frozen."""

        def no_train(model, mode=True):
            return model

        if freezing:
            self.model.eval()
            self.model.train = no_train
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train = type(self.model).train
            self.model.train()

    def unfreeze(self):
        """Like `freeze(False)`."""
        self.freeze(freezing=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        reduction: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
            torch.mean, dim=0
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

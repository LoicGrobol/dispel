import pathlib

from typing import List, Optional

import click
import click_pathlib
import more_itertools
import torch
import tqdm

from dispel import Vectorizer


def read_conll(f: pathlib.Path, word_col: int = 0) -> List[List[str]]:
    res = []
    current_sent: List[str] = []
    with open(f) as in_stream:
        for l in in_stream:
            if l.isspace():
                if current_sent:
                    res.append(current_sent)
                    current_sent = []
            elif l.startswith("#"):
                continue
            else:
                current_sent.append(l.strip().split()[word_col])
    if current_sent:
        res.append(current_sent)
    return res


@click.command()
@click.argument(
    "conll-file",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.argument(
    "out-file", type=click_pathlib.Path(resolve_path=True, dir_okay=False),
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help=("Number of sentences in a processing batch"),
)
@click.option(
    "--model",
    type=str,
    default="roberta-base",
    help="A huggingface transformers pretrained model",
    metavar="NAME_OR_PATH",
)
def main(
    batch_size: int, conll_file: pathlib.Path, model: str, out_file: pathlib.Path,
):
    vecto = Vectorizer.from_pretrained(model)
    vecto.freeze()
    sents = read_conll(conll_file)
    with open(out_file, "w") as out_stream:
        pbar = tqdm.tqdm(
            more_itertools.chunked(sents, batch_size),
            desc="Vectorizing",
            leave=False,
            total=len(sents) // batch_size,
            unit="batch",
        )
        for chunk in pbar:
            with torch.no_grad():
                chunk_vecs = vecto.vectorize(chunk)
            for sent, sent_vecs in zip(chunk, chunk_vecs):
                for w, v in zip(sent, sent_vecs.last_hidden_state.unbind(0)):
                    w_line = " ".join([w, *(str(c) for c in v.tolist())])
                    out_stream.write(w_line)
                    out_stream.write("\n")
                out_stream.write("\n")


if __name__ == "__main__":
    main()

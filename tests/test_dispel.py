from dispel import Vectorizer


def test_vectorizer():
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
    b = v.vectorize(sents)
    assert len(b) == len(sents)
    for so, se in zip(sents, b):
        assert se.last_hidden_state.shape[:-1] == torch.Size([len(so)])
        assert all(l.shape[:-1] == torch.Size([len(so)]) for l in se.hidden_states)
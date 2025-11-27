from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", max_len=16):
        self.max_len = max_len
        self.unk = unk_token
        self.pad = pad_token

        if vocab is None:
            words = ["t-shirt", "trouser", "pullover", "dress", "coat",
                     "sandal", "shirt", "sneaker", "bag", "ankleboot",
                     "shoe", "jacket", "boots", "sleeveless", "shorts"]
            vocab = [pad_token, unk_token] + words

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text):
        """
        text: string
        return: list[int] length max_len
        """
        tokens = text.lower().split()
        ids = []
        for t in tokens[:self.max_len]:
            ids.append(self.word2idx.get(t, self.word2idx[self.unk]))
        # pad
        while len(ids) < self.max_len:
            ids.append(self.word2idx[self.pad])
        return ids

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, self.unk) for i in ids)

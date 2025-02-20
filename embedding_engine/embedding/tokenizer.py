from gensim.parsing.preprocessing import remove_stopwords

from embedding_engine.config import Config


class Tokenizer:
    def tokenize(self, phrase: str) -> list[str]:
        return phrase.split()


class TokenizerV2(Tokenizer):
    """Version of the tokenizer that removes stopwords and duplicates."""

    def tokenize(self, phrase: str) -> list[str]:
        phrase = remove_stopwords(phrase)
        words = list(set(phrase.split()))
        return words


def get_tokenizer():
    match Config.tokenizer:
        case "V1":
            return Tokenizer()
        case "V2":
            return TokenizerV2()
        case _:
            raise ValueError(f"Tokenizer type '{Config.tokenizer}' does not exist!")

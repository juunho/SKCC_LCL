from konlpy.tag import Mecab

class Preprocessing:
    def __init__(self, text):
        self.text = text
        self.mecab = Mecab()

    def extract_mecab_nouns(self, sentences):
        return [self.mecab.nouns(sentence) for sentence in sentences]

    def preprocess(self):
        sentences = self.text.split('. ')  # Split text into sentences (assuming sentences end with '. ')
        quoted_sentences = ['"' + sentence + '"' for sentence in sentences]  # Add quotes to each sentence
        # Extract nouns using Mecab
        mecab_nouns = self.extract_mecab_nouns(quoted_sentences)
        # Split texts based on Mecab nouns
        return mecab_nouns
    

    
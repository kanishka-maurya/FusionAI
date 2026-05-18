from sentence_transformers import CrossEncoder


class SemanticAlignment:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def compare(self, text1, text2):

        score = self.model.predict([
            [text1, text2]
        ])

        return float(score[0])


semantic_alignment = SemanticAlignment()
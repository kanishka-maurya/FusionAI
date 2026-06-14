class SemanticAlignment:

    def __init__(self):
        self.model = None

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        return self.model

    def compare(self, text1, text2):

        score = self._get_model().predict([
            [text1, text2]
        ])

        return float(score[0])


semantic_alignment = SemanticAlignment()

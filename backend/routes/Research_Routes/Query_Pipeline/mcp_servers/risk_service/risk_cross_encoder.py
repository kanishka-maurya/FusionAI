from sentence_transformers import CrossEncoder


class RiskCrossEncoder:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def score(self, text1, text2):

        combined = [[text1, text2]]

        score = self.model.predict(combined)

        return float(score[0])


risk_cross_encoder = RiskCrossEncoder()
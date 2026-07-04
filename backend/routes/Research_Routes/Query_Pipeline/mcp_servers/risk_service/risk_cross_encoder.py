import os
import re
from threading import Lock


def _env_enabled(name, default="false"):

    return (
        os.getenv(name, default)
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _tokens(text):

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            (text or "").lower()
        )
    )


def _fast_similarity(text1, text2):

    tokens1 = _tokens(text1)
    tokens2 = _tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    overlap = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    smaller = min(len(tokens1), len(tokens2))

    jaccard = overlap / union
    containment = overlap / smaller

    return round(
        (0.65 * jaccard) + (0.35 * containment),
        4
    )


class RiskCrossEncoder:

    def __init__(self):
        self.model = None
        self._model_lock = Lock()
        self._predict_lock = Lock()
        self.use_cross_encoder = _env_enabled(
            "RESEARCH_USE_CROSS_ENCODER",
            "false"
        )

    def _get_model(self):
        if not self.use_cross_encoder:
            return None

        if self.model is None:
            with self._model_lock:

                if self.model is None:
                    print(
                        "\n[RISK CROSS ENCODER] Loading model "
                        "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )

                    from sentence_transformers import CrossEncoder

                    self.model = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        local_files_only=True
                    )

                    print(
                        "\n[RISK CROSS ENCODER] Model loaded"
                    )
        return self.model

    def score(self, text1, text2):

        if not self.use_cross_encoder:
            score = _fast_similarity(
                text1,
                text2
            )

            print(
                "\n[RISK FAST SCORER] "
                f"Lexical score={score}"
            )

            return score

        combined = [[text1, text2]]

        print(
            "\n[RISK CROSS ENCODER] Predict start"
        )

        try:

            model = self._get_model()

            if model is None:
                return _fast_similarity(text1, text2)

            with self._predict_lock:
                score = model.predict(combined)

        except Exception as e:

            fallback_score = _fast_similarity(
                text1,
                text2
            )

            print(
                "\n[RISK CROSS ENCODER FALLBACK]",
                repr(e),
                f"lexical_score={fallback_score}"
            )

            return fallback_score

        print(
            "\n[RISK CROSS ENCODER] Predict done"
        )

        return float(score[0])


risk_cross_encoder = RiskCrossEncoder()

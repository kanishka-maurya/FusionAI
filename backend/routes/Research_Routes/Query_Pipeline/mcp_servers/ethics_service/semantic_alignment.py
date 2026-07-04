from backend.routes.Research_Routes.Query_Pipeline.mcp_servers.risk_service.risk_cross_encoder import (
    risk_cross_encoder
)


class SemanticAlignment:

    def compare(self, text1, text2):

        print(
            "\n[SEMANTIC ALIGNMENT] Scoring pair"
        )

        return risk_cross_encoder.score(
            text1,
            text2
        )


semantic_alignment = SemanticAlignment()

import logging
import os
import sys
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from zep_cloud.client import Zep
from backend.core.exceptions import CustomException
from zep_crewai import ZepUserStorage
from crewai.memory.external.external_memory import ExternalMemory
from services.research_service.generation.generation import RAGResult

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    user_query: str
    assistant_response: str
    sources_used: List[Dict[str, Any]]
    timestamp: str
    session_id: str

class NotebookMemoryLayer:

    def __init__(
        self,
        user_id: str,
        session_id: str,
        zep_api_key: Optional[str] = None,
        mode: str = "summary",
        indexing_wait_time: int = 10,
        create_new_session: bool = False
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.indexing_wait_time = indexing_wait_time

        self.zep_client = Zep(api_key=zep_api_key or os.getenv("ZEP_API_KEY"))

        self._setup_user_and_session(create_new_session)

        self.user_storage = ZepUserStorage(
            client=self.zep_client,
            user_id=self.user_id,
            thread_id=self.session_id,
            mode=mode,
        )

        self.external_memory = ExternalMemory(storage=self.user_storage)

        logger.info(f"Memory initialized for user={user_id}, session={session_id}")


    def _setup_user_and_session(self, create_new_session: bool):
        try:
            # Ensure user
            try:
                self.zep_client.user.get(self.user_id)
            except:
                self.zep_client.user.add(user_id=self.user_id)

            # Session handling
            if create_new_session:
                try:
                    self.zep_client.thread.delete(self.session_id)
                except:
                    pass

                self.zep_client.thread.create(
                    thread_id=self.session_id,
                    user_id=self.user_id
                )
            else:
                try:
                    self.zep_client.thread.get(self.session_id)
                except:
                    self.zep_client.thread.create(
                        thread_id=self.session_id,
                        user_id=self.user_id
                    )

        except Exception as e:
            raise CustomException(e, sys)


    def save_conversation_turn(
        self,
        rag_result: RAGResult,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None
    ):
        try:
            # USER MESSAGE
            user_meta = {
                "type": "message",
                "role": "user",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                **(user_metadata or {})
            }

            self.external_memory.save(
                rag_result.query,
                metadata=user_meta
            )

            # ASSISTANT MESSAGE
            assistant_meta = {
                "type": "message",
                "role": "assistant",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "sources_count": len(rag_result.sources_used),
                "retrieval_count": rag_result.retrieval_count,
                "sources_summary": self._create_sources_summary(rag_result.sources_used),
                **(assistant_metadata or {})
            }

            self.external_memory.save(
                rag_result.response,
                metadata=assistant_meta
            )

            # SAVE SOURCE CONTEXT
            self._save_source_context(rag_result.sources_used)

        except Exception as e:
            raise CustomException(e, sys)


    def _create_sources_summary(self, sources_used: List[Dict[str, Any]]) -> str:
        if not sources_used:
            return "No sources used"

        source_files = list(set(
            s.get('source_file') or 'Unknown' for s in sources_used
        ))

        source_types = list(set(
            s.get('source_type') or 'unknown' for s in sources_used
        ))

        summary = f"{len(source_files)} files ({', '.join(source_types)}): {', '.join(source_files[:3])}"

        if len(source_files) > 3:
            summary += f" and {len(source_files) - 3} more"

        return summary


    def _save_source_context(self, sources_used: List[Dict[str, Any]]):
        if not sources_used:
            return

        source_context = {
            "referenced_documents": [],
            "document_types": set()
        }

        for source in sources_used:
            doc_info = {
                "file": source.get('source_file', 'Unknown'),
                "type": source.get('source_type', 'unknown'),
                "page": source.get('page_number'),
                "relevance": source.get('relevance_score', 0)
            }

            source_context["referenced_documents"].append(doc_info)
            source_context["document_types"].add(doc_info["type"])

        source_context["document_types"] = list(source_context["document_types"])

        self.external_memory.save(
            f"Document sources referenced: {source_context}",
            metadata={
                "type": "source_context",
                "category": "document_usage",
                "session_id": self.session_id
            }
        )


    def save_user_preferences(self, preferences: Dict[str, Any]):
        self.external_memory.save(
            f"User preferences: {preferences}",
            metadata={
                "type": "preferences",
                "category": "user_settings",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
        )


    def get_conversation_context(self) -> str:
        try:
            memory = self.zep_client.thread.get_user_context(
                thread_id=self.session_id
            )
            return memory.context if memory.context else ""
        except:
            return ""

    def get_last_n_turns_memory(self, n: int = 4):
        try:
            response = self.zep_client.thread.get(thread_id=self.session_id)
            messages = response.messages or []

            last_messages = messages[-2 * n:]

            return [
                {"role": m.role, "message": m.content}
                for m in last_messages
            ]
        except:
            return []

    def get_relevant_memory(self, query: str, limit: int = 5):
        try:
            results = self.zep_client.graph.search(
                user_id=self.user_id,
                thread_id=self.session_id,  
                query=query,
                scope="episodes"
            )

            return [
                {
                    "content": ep.content,
                    "role": ep.role_type,
                    "score": getattr(ep, "score", 0)
                }
                for ep in results.episodes[:limit]
            ]

        except:
            return []


    def build_memory_context(self, user_query: str, limit: int = 5):

        summary = self.get_session_summary()
        recent = self.get_last_n_turns_memory(n=4)
        relevant = self.get_relevant_memory(user_query, limit)
        context_summary = self.get_conversation_context()

        # FORMAT FOR LLM
        formatted_recent = "\n".join(
            [f"{r['role']}: {r['message']}" for r in recent]
        ) if recent else "No recent chats"

        formatted_relevant = "\n".join(
            [f"- {m['content']}" for m in relevant]
        ) if relevant else "No relevant memory"

        formatted = f"""
SESSION SUMMARY:
Total Messages: {summary.get('total_messages', 0)}

RECENT:
{formatted_recent}

RELEVANT MEMORY:
{formatted_relevant}

SUMMARY:
{context_summary}
"""

        return {
            "raw": {
                "summary": summary,
                "recent": recent,
                "relevant": relevant
            },
            "formatted": formatted  
        }
    def get_session_summary(self):
        try:
            response = self.zep_client.thread.get(thread_id=self.session_id)
            messages = response.messages or []

            return {
                "total_messages": len(messages),
                "user_messages": len([m for m in messages if m.role == "user"]),
                "assistant_messages": len([m for m in messages if m.role == "assistant"]),
            }
        except:
            return {}

    def wait_for_indexing(self):
        time.sleep(self.indexing_wait_time)

    def clear_session(self):
        self.zep_client.thread.delete(self.session_id)
        self.zep_client.thread.create(
            thread_id=self.session_id,
            user_id=self.user_id
        )
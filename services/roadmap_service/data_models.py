# models.py
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
 
 
class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
 
 
class GenerateRoadmapRequest(BaseModel):
    topic: str
    level: DifficultyLevel = DifficultyLevel.BEGINNER
    user_id: Optional[str] = None
 
 
class NodePosition(BaseModel):
    x: float
    y: float
 
 
class Resource(BaseModel):
    type: str        # "video" | "article" | "paper"
    title: str
    url: str
 
 
class SubTopic(BaseModel):
    title: str
    explanation: str
    code_example: Optional[str] = None
    key_takeaway: str
 
 
class NodeContent(BaseModel):
    summary: str
    estimated_time: str
    what_you_will_learn: List[str]
    topics: List[SubTopic]
    common_misconceptions: List[str]
    resources: List[Resource]
    practice_questions: List[dict]
 
 
class RoadmapNode(BaseModel):
    id: str
    title: str
    type: str           # "required" | "optional" | "project"
    level: str
    status: str         # "locked" | "unlocked"
    dependencies: List[str]
    position: NodePosition
    content_generated: bool = False
 
 
class RoadmapResponse(BaseModel):
    roadmap_id: str
    user_id: str
    title: str
    topic: str
    description: str
    total_nodes: int
    nodes: List[RoadmapNode]

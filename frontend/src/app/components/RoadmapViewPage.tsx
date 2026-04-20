import { useState, useEffect } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import {
  ArrowLeft,
  CheckCircle,
  Lock,
  Circle,
  Sparkles,
  BookOpen,
  Clock,
  Target,
} from "lucide-react";
import { NodeContentModal } from "./NodeContentModal";
import { supabase } from "../contexts/AuthContext";

export interface RoadmapNode {
  node_id: string;
  roadmap_id: string;
  title: string;
  type?: string | null;
  level: string;
  status: string;
  dependencies?: string[] | null;
  position_x?: number | null;
  position_y?: number | null;
  content_generated?: boolean | null;
  raw_content?: any | null;
  created_at?: string | null;
}

interface RoadmapData {
  roadmap_id: string;
  user_id?: string;
  title: string;
  topic: string;
  description: string;
  total_nodes: number;
  created_at?: string;
}

export function RoadmapViewPage() {
  const navigate = useNavigate();
  const { roadmapId } = useParams();
  const location = useLocation();
  const { user } = useAuth();

  const [roadmapData, setRoadmapData] = useState<RoadmapData | null>(null);
  const [nodes, setNodes] = useState<RoadmapNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<RoadmapNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadRoadmap();
  }, [roadmapId]);

  const loadRoadmap = async () => {
    setIsLoading(true);
    try {
      const { data: roadmapInfo } = await supabase
        .from("roadmaps")
        .select("*")
        .eq("roadmap_id", roadmapId)
        .single();

      if (roadmapInfo) setRoadmapData(roadmapInfo);
      const { data: nodesData } = await supabase
        .from("nodes")
        .select("*")
        .eq("roadmap_id", roadmapId);

      const normalizedNodes: RoadmapNode[] = (nodesData || []).map(
        (n: any) => ({
          node_id: n.node_id,
          roadmap_id: n.roadmap_id,
          title: n.title,
          type: n.type,
          level: n.level,
          status: n.status,
          dependencies: Array.isArray(n.dependencies)
            ? n.dependencies
            : typeof n.dependencies === "string"
              ? JSON.parse(n.dependencies || "[]")
              : [],
          position_x: Number(n.position_x ?? 50),
          position_y: Number(n.position_y ?? 50),

          content_generated: n.content_generated,
          raw_content: n.raw_content,
          created_at: n.created_at,
        }),
      );

      setNodes(normalizedNodes);
      // Mock data based on the topic from navigation state
      const { topic = "Machine Learning", level = "beginner" } =
        location.state || {};

      const mockRoadmapData: RoadmapData = {
        roadmap_id: roadmapId || "",
        user_id: user?.id,
        title: `${topic} Learning Path`,
        topic: topic,
        description: `A comprehensive ${level}-level roadmap to master ${topic}`,
        total_nodes: 8,
      };

      const mockNodes: RoadmapNode[] = [
        {
          node_id: "node_1",
          roadmap_id: roadmapId || "",
          title: "Introduction & Fundamentals",
          type: "concept",
          level: "beginner",
          status: "unlocked",
          dependencies: [],
          position_x: 50,
          position_y: 10,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_2",
          roadmap_id: roadmapId || "",
          title: "Mathematical Foundations",
          type: "concept",
          level: "beginner",
          status: "locked",
          dependencies: ["node_1"],
          position_x: 25,
          position_y: 30,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_3",
          roadmap_id: roadmapId || "",
          title: "Programming Basics",
          type: "concept",
          level: "beginner",
          status: "locked",
          dependencies: ["node_1"],
          position_x: 75,
          position_y: 30,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_4",
          roadmap_id: roadmapId || "",
          title: "Data Preprocessing",
          type: "skill",
          level: "intermediate",
          status: "locked",
          dependencies: ["node_2", "node_3"],
          position_x: 35,
          position_y: 50,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_5",
          roadmap_id: roadmapId || "",
          title: "Supervised Learning",
          type: "concept",
          level: "intermediate",
          status: "locked",
          dependencies: ["node_4"],
          position_x: 20,
          position_y: 70,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_6",
          roadmap_id: roadmapId || "",
          title: "Unsupervised Learning",
          type: "concept",
          level: "intermediate",
          status: "locked",
          dependencies: ["node_4"],
          position_x: 50,
          position_y: 70,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_7",
          roadmap_id: roadmapId || "",
          title: "Neural Networks",
          type: "concept",
          level: "advanced",
          status: "locked",
          dependencies: ["node_5", "node_6"],
          position_x: 35,
          position_y: 90,
          content_generated: false,
          raw_content: null,
        },
        {
          node_id: "node_8",
          roadmap_id: roadmapId || "",
          title: "Advanced Topics & Projects",
          type: "project",
          level: "advanced",
          status: "locked",
          dependencies: ["node_7"],
          position_x: 50,
          position_y: 110,
          content_generated: false,
          raw_content: null,
        },
      ];

      await new Promise((resolve) => setTimeout(resolve, 800));
      setRoadmapData(mockRoadmapData);
      setNodes(mockNodes);
    } catch (error) {
      console.error("Failed to load roadmap:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = (node: RoadmapNode) => {
    if (node.status === "locked") return;
    setSelectedNode(node);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case "unlocked":
        return <Circle className="w-5 h-5 text-blue-600" />;
      case "locked":
        return <Lock className="w-5 h-5 text-gray-400" />;
      default:
        return <Circle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "beginner":
        return "bg-green-100 text-green-700 border-green-300";
      case "intermediate":
        return "bg-yellow-100 text-yellow-700 border-yellow-300";
      case "advanced":
        return "bg-red-100 text-red-700 border-red-300";
      default:
        return "bg-gray-100 text-gray-700 border-gray-300";
    }
  };

  const stats = {
    total: roadmapData?.total_nodes || nodes.length,
    unlocked: nodes.filter((n) => n.status === "unlocked").length,
    completed: nodes.filter((n) => n.status === "completed").length,
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Generating your roadmap...</p>
        </div>
      </div>
    );
  }

  if (!roadmapData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Roadmap not found</p>
          <button
            onClick={() => navigate("/roadmap")}
            className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Back to Roadmap
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/roadmap")}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex items-center gap-3 flex-1">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  {roadmapData.title}
                </h1>
                <p className="text-sm text-gray-600">
                  {roadmapData.description}
                </p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-gray-600" />
              <span className="text-sm text-gray-600">
                <span className="font-semibold text-gray-900">
                  {stats.total}
                </span>{" "}
                Topics
              </span>
            </div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-blue-600" />
              <span className="text-sm text-gray-600">
                <span className="font-semibold text-blue-600">
                  {stats.unlocked}
                </span>{" "}
                Unlocked
              </span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-600">
                <span className="font-semibold text-green-600">
                  {stats.completed}
                </span>{" "}
                Completed
              </span>
            </div>
            <div className="flex-1">
              <div className="w-full max-w-md h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-600 to-purple-500 rounded-full transition-all"
                  style={{ width: `${(stats.completed / stats.total) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Roadmap Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="relative">
          {/* SVG for connecting lines */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 0 }}
          >
            {nodes.map((node) =>
              node.dependencies?.map((depId) => {
                const depNode = nodes.find((n) => n.node_id === depId);
                if (
                  !depNode ||
                  !depNode.position_x ||
                  !depNode.position_y ||
                  !node.position_x ||
                  !node.position_y
                )
                  return null;

                const x1 = `${depNode.position_x}%`;
                const y1 = `${depNode.position_y}%`;
                const x2 = `${node.position_x}%`;
                const y2 = `${node.position_y}%`;

                return (
                  <line
                    key={`${depId}-${node.node_id}`}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#e5e7eb"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                  />
                );
              }),
            )}
          </svg>

          {/* Nodes */}
          <div className="relative" style={{ minHeight: "120vh" }}>
            {nodes.map((node) => (
              <div
                key={node.node_id}
                className="absolute"
                style={{
                  left: `${node.position_x}%`,
                  top: `${node.position_y}%`,
                  transform: "translate(-50%, -50%)",
                  zIndex: 10,
                }}
              >
                <div
                  onClick={() => handleNodeClick(node)}
                  className={`
                    w-64 bg-white rounded-xl border-2 p-5 shadow-lg transition-all cursor-pointer
                    ${
                      node.status === "locked"
                        ? "border-gray-300 opacity-60 cursor-not-allowed"
                        : node.status === "completed"
                          ? "border-green-400 hover:shadow-xl hover:scale-105"
                          : "border-blue-400 hover:shadow-xl hover:scale-105"
                    }
                  `}
                >
                  <div className="flex items-start gap-3 mb-3">
                    <div className="flex-shrink-0 mt-1">
                      {getStatusIcon(node.status)}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 mb-1">
                        {node.title}
                      </h3>
                      {node.type && (
                        <p className="text-xs text-gray-500 capitalize mb-1">
                          {node.type}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium border ${getLevelColor(node.level)}`}
                    >
                      {node.level}
                    </span>
                    {node.status === "locked" &&
                      node.dependencies &&
                      node.dependencies.length > 0 && (
                        <span className="text-xs text-gray-500">
                          {node.dependencies.length} prerequisite
                          {node.dependencies.length > 1 ? "s" : ""}
                        </span>
                      )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Node Content Modal */}
      {/* {selectedNode && roadmapData && (
        <NodeContentModal
          node={selectedNode}
          roadmapId={roadmapData.roadmap_id}
          onClose={() => setSelectedNode(null)}
        />
      )} */}
    </div>
  );
}

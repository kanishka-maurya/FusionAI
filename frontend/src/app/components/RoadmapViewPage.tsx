import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth, supabase } from "../contexts/AuthContext";
import {
  ArrowLeft,
  CheckCircle,
  Lock,
  Circle,
  Sparkles,
  BookOpen,
  Target,
  Loader2,
  Route,
} from "lucide-react";
import { NodeContentModal } from "./NodeContentModal";

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

const API_BASE = "http://localhost:8000";

export function RoadmapViewPage() {
  const navigate = useNavigate();
  const { roadmapId } = useParams();
  const { user } = useAuth();

  const [roadmapData, setRoadmapData] = useState<RoadmapData | null>(null);
  const [nodes, setNodes] = useState<RoadmapNode[]>([]);
  const [selectedNode, setSelectedNode] = useState<RoadmapNode | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    loadRoadmap();
  }, [roadmapId]);

  const getToken = async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return session?.access_token;
  };

  const normalizeDependencies = (value: any): string[] => {
    if (Array.isArray(value)) return value;
    if (typeof value === "string") {
      try {
        return JSON.parse(value || "[]");
      } catch {
        return [];
      }
    }
    return [];
  };

  const loadRoadmap = async () => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE}/api/roadmap/${roadmapId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to load roadmap");

      const normalizedNodes: RoadmapNode[] = (data.nodes || []).map((node: any) => ({
        node_id: node.node_id,
        roadmap_id: node.roadmap_id,
        title: node.title,
        type: node.type,
        level: node.level,
        status: node.status,
        dependencies: normalizeDependencies(node.dependencies),
        position_x: Number(node.position_x ?? 400),
        position_y: Number(node.position_y ?? 50),
        content_generated: node.content_generated,
        raw_content: node.raw_content,
        created_at: node.created_at,
      }));

      setRoadmapData({
        roadmap_id: data.roadmap_id,
        user_id: data.user_id,
        title: data.title,
        topic: data.topic,
        description: data.description,
        total_nodes: data.total_nodes || normalizedNodes.length,
        created_at: data.created_at,
      });
      setNodes(normalizedNodes);
    } catch (error: any) {
      console.error("Failed to load roadmap:", error);
      setErrorMessage(error.message || "Roadmap not found");
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = (node: RoadmapNode) => {
    if (node.status === "locked") return;
    setSelectedNode(node);
  };

  const isDone = (status: string) => status === "done" || status === "completed";

  const getStatusIcon = (status: string) => {
    if (isDone(status)) return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    if (status === "unlocked" || status === "in_progress") return <Circle className="w-4 h-4 text-blue-400" />;
    return <Lock className="w-4 h-4 text-slate-500" />;
  };

  const getNodeTone = (node: RoadmapNode) => {
    if (node.status === "locked") return "border-white/5 bg-white/[0.02] opacity-60";
    if (isDone(node.status)) return "border-emerald-500/30 bg-emerald-500/10";
    return "border-blue-500/30 bg-blue-500/10 hover:bg-blue-500/15";
  };

  const stats = {
    total: roadmapData?.total_nodes || nodes.length,
    unlocked: nodes.filter((n) => n.status === "unlocked" || n.status === "in_progress").length,
    completed: nodes.filter((n) => isDone(n.status)).length,
  };
  const progress = stats.total ? Math.round((stats.completed / stats.total) * 100) : 0;
  const maxX = Math.max(900, ...nodes.map((n) => Number(n.position_x || 0) + 220));
  const maxY = Math.max(640, ...nodes.map((n) => Number(n.position_y || 0) + 180));

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0d0e1b] flex items-center justify-center text-slate-200">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Loading route...
          </p>
        </div>
      </div>
    );
  }

  if (!roadmapData) {
    return (
      <div className="min-h-screen bg-[#0d0e1b] flex items-center justify-center text-slate-200">
        <div className="text-center">
          <p className="text-sm text-slate-400">{errorMessage || "Roadmap not found"}</p>
          <button
            onClick={() => navigate("/roadmap")}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-xl text-xs font-bold"
          >
            Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0d0e1b] text-slate-100 font-sans antialiased overflow-hidden">
      <header className="bg-[#111322]/60 backdrop-blur-md border-b border-white/5 sticky top-0 z-40">
        <div className="px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 min-w-0">
            <button
              onClick={() => navigate("/roadmap")}
              className="p-2 bg-white/[0.02] border border-white/5 hover:bg-white/5 text-slate-400 hover:text-white rounded-xl transition-all shrink-0"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-violet-500 rounded-xl flex items-center justify-center border border-blue-400/20 shadow-lg shadow-blue-500/10 shrink-0">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-sm font-bold text-white uppercase tracking-wider truncate">
                {roadmapData.title}
              </h1>
              <p className="text-[11px] text-slate-400 mt-0.5 truncate">
                {roadmapData.description}
              </p>
            </div>
          </div>
          <div className="hidden md:block text-right">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">User</p>
            <p className="text-xs text-slate-300">{user?.name || user?.email}</p>
          </div>
        </div>
      </header>

      <div className="border-b border-white/5 bg-[#111322]/40">
        <div className="px-6 py-4 grid grid-cols-1 md:grid-cols-[1fr_280px] gap-4 items-center">
          <div className="flex flex-wrap items-center gap-4">
            <div className="inline-flex items-center gap-2 text-xs text-slate-400">
              <Target className="w-4 h-4 text-blue-400" />
              <span><b className="text-white">{stats.total}</b> topics</span>
            </div>
            <div className="inline-flex items-center gap-2 text-xs text-slate-400">
              <BookOpen className="w-4 h-4 text-violet-400" />
              <span><b className="text-white">{stats.unlocked}</b> unlocked</span>
            </div>
            <div className="inline-flex items-center gap-2 text-xs text-slate-400">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span><b className="text-white">{stats.completed}</b> completed</span>
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between text-[10px] text-slate-500 mb-1">
              <span>Progress</span>
              <span>{progress}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-emerald-400 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="mx-6 mt-4 rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          {errorMessage}
        </div>
      )}

      <main className="h-[calc(100vh-145px)] overflow-auto p-6">
        <div
          className="relative rounded-2xl border border-white/5 bg-[#111322]/50 shadow-2xl shadow-black/20"
          style={{ width: maxX, height: maxY }}
        >
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {nodes.flatMap((node) =>
              (node.dependencies || []).map((depId) => {
                const depNode = nodes.find((n) => n.node_id === depId);
                if (!depNode) return null;
                return (
                  <line
                    key={`${depId}-${node.node_id}`}
                    x1={Number(depNode.position_x || 0)}
                    y1={Number(depNode.position_y || 0)}
                    x2={Number(node.position_x || 0)}
                    y2={Number(node.position_y || 0)}
                    stroke="rgba(148,163,184,0.25)"
                    strokeWidth="2"
                    strokeDasharray="6 8"
                  />
                );
              })
            )}
          </svg>

          {nodes.map((node) => (
            <button
              key={node.node_id}
              onClick={() => handleNodeClick(node)}
              className={`absolute w-64 text-left rounded-2xl border p-4 transition-all ${getNodeTone(node)} ${
                node.status === "locked" ? "cursor-not-allowed" : "hover:scale-[1.02] hover:border-blue-400/50"
              }`}
              style={{
                left: Number(node.position_x || 0),
                top: Number(node.position_y || 0),
                transform: "translate(-50%, -50%)",
              }}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">{getStatusIcon(node.status)}</div>
                <div className="min-w-0">
                  <h3 className="text-sm font-bold text-white leading-snug line-clamp-2">
                    {node.title}
                  </h3>
                  <div className="flex flex-wrap items-center gap-2 mt-3">
                    <span className="px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide bg-white/5 text-slate-300 border border-white/5">
                      {node.level}
                    </span>
                    {node.type && (
                      <span className="px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wide bg-blue-500/10 text-blue-300 border border-blue-500/20">
                        {node.type}
                      </span>
                    )}
                  </div>
                  {node.status === "locked" && !!node.dependencies?.length && (
                    <p className="text-[10px] text-slate-500 mt-3">
                      {node.dependencies.length} prerequisite{node.dependencies.length > 1 ? "s" : ""}
                    </p>
                  )}
                </div>
              </div>
            </button>
          ))}

          <div className="absolute left-5 top-5 inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-black/20 border border-white/5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <Route className="w-3.5 h-3.5 text-blue-400" />
            Graph
          </div>
        </div>
      </main>

      {selectedNode && roadmapData && (
        <NodeContentModal
          node={selectedNode}
          roadmapId={roadmapData.roadmap_id}
          onClose={() => setSelectedNode(null)}
          onStatusChange={loadRoadmap}
        />
      )}
    </div>
  );
}

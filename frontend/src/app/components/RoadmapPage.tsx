import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth, supabase } from "../contexts/AuthContext";
import { useRoadmap } from "../contexts/RoadmapContext";
import {
  ArrowLeft,
  GraduationCap,
  Route,
  Sparkles,
  LogOut,
  Plus,
  Loader2,
  Layers3,
  Target,
  Clock3,
  ChevronRight,
} from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

type Roadmap = {
  roadmap_id: string;
  title: string;
  description: string;
  topic: string;
  total_nodes: number;
  created_at?: string;
};

const API_BASE = "http://localhost:8000";

export function RoadmapPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { setRoadmap } = useRoadmap();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("beginner");
  const [isGenerating, setIsGenerating] = useState(false);
  const [roadmaps, setRoadmaps] = useState<Roadmap[]>([]);
  const [loadingRoadmaps, setLoadingRoadmaps] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  const getToken = async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return session?.access_token;
  };

  const fetchRoadmaps = async () => {
    setLoadingRoadmaps(true);
    setErrorMessage("");
    try {
      const token = await getToken();
      const res = await fetch(`${API_BASE}/api/roadmap/user`, {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to fetch roadmaps");
      setRoadmaps(data.roadmaps || []);
    } catch (err: any) {
      console.error("Failed to fetch roadmaps:", err);
      setErrorMessage(err.message || "Unable to load Pathfinder routes.");
    } finally {
      setLoadingRoadmaps(false);
    }
  };

  useEffect(() => {
    fetchRoadmaps();
  }, []);

  const handleGenerateRoadmap = async () => {
    if (!topic.trim()) return;

    setIsGenerating(true);
    setErrorMessage("");
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      const response = await fetch(`${API_BASE}/api/roadmap/generate`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          topic,
          level,
          user_id: session?.user?.id,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate roadmap");
      }

      setRoadmap(data);
      setIsCreateDialogOpen(false);
      setTopic("");
      setLevel("beginner");
      navigate(`/roadmap/${data.roadmap_id}`);
    } catch (error: any) {
      console.error("Failed to generate roadmap:", error);
      setErrorMessage(error.message || "Pathfinder generation failed.");
    } finally {
      setIsGenerating(false);
    }
  };

  const formatDate = (value?: string) => {
    if (!value) return "New route";
    return new Date(value).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="min-h-screen bg-[#0d0e1b] text-slate-100 font-sans antialiased">
      <header className="bg-[#111322]/60 backdrop-blur-md border-b border-white/5 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/dashboard")}
              className="p-2 bg-white/[0.02] border border-white/5 hover:bg-white/5 text-slate-400 hover:text-white rounded-xl transition-all"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-violet-500 rounded-xl flex items-center justify-center border border-blue-400/20 shadow-lg shadow-blue-500/10">
                <GraduationCap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-white uppercase tracking-wider">
                  Pathfinder AI
                </h1>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Learning routes
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-3 px-3 py-1.5 rounded-xl bg-white/[0.02] border border-white/5">
              <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-600 flex items-center justify-center text-white text-xs font-bold overflow-hidden ring-1 ring-white/10">
                {user?.avatar ? (
                  <img src={user.avatar} alt={user.name} className="w-full h-full object-cover" />
                ) : (
                  <span>{user?.name?.charAt(0) || "S"}</span>
                )}
              </div>
              <div className="text-right">
                <p className="text-[11px] font-semibold text-white leading-tight">
                  {user?.name || user?.email}
                </p>
              </div>
            </div>
            <button
              onClick={logout}
              className="p-2 md:px-4 md:py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-[#111322] hover:bg-[#16192e] rounded-xl transition-all border border-white/10"
            >
              <LogOut className="w-4 h-4 md:hidden" />
              <span className="hidden md:inline">Sign out</span>
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 mb-8">
          <section className="border border-white/5 bg-[#111322]/70 rounded-2xl p-6 shadow-2xl shadow-black/20">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20 text-[10px] font-bold uppercase tracking-wider text-blue-300 mb-4">
                  <Sparkles className="w-3.5 h-3.5" />
                  Pathfinder
                </div>
                <h2 className="text-3xl font-black tracking-tight text-white max-w-2xl">
                  Build a structured learning path.
                </h2>
                <p className="text-sm text-slate-400 mt-3 max-w-2xl leading-relaxed">
                  Generate a route from any topic and level.
                </p>
              </div>
              <button
                onClick={() => setIsCreateDialogOpen(true)}
                className="shrink-0 inline-flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-violet-500 text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 transition-all shadow-lg shadow-blue-500/15"
              >
                <Plus className="w-4 h-4" />
                New Route
              </button>
            </div>
          </section>

          <aside className="grid grid-cols-3 lg:grid-cols-1 gap-3">
            <div className="bg-[#111322]/70 border border-white/5 rounded-2xl p-4">
              <Route className="w-4 h-4 text-blue-400 mb-3" />
              <p className="text-2xl font-black text-white">{roadmaps.length}</p>
              <p className="text-[11px] text-slate-400 font-medium">Routes</p>
            </div>
            <div className="bg-[#111322]/70 border border-white/5 rounded-2xl p-4">
              <Layers3 className="w-4 h-4 text-violet-400 mb-3" />
              <p className="text-2xl font-black text-white">
                {roadmaps.reduce((sum, r) => sum + (r.total_nodes || 0), 0)}
              </p>
              <p className="text-[11px] text-slate-400 font-medium">Nodes</p>
            </div>
            <div className="bg-[#111322]/70 border border-white/5 rounded-2xl p-4">
              <Target className="w-4 h-4 text-emerald-400 mb-3" />
              <p className="text-2xl font-black text-white">AI</p>
              <p className="text-[11px] text-slate-400 font-medium">Guided</p>
            </div>
          </aside>
        </div>

        {errorMessage && (
          <div className="mb-6 rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {errorMessage}
          </div>
        )}

        <section>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
                Learning Routes
              </h3>
              <p className="text-[11px] text-slate-500 mt-1">
                Continue or create a route.
              </p>
            </div>
          </div>

          {loadingRoadmaps ? (
            <div className="h-64 flex items-center justify-center rounded-2xl border border-white/5 bg-[#111322]/50">
              <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
            </div>
          ) : roadmaps.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center rounded-2xl border border-white/5 bg-[#111322]/50 text-center">
              <GraduationCap className="w-10 h-10 text-slate-500 mb-3" />
              <p className="text-sm font-bold text-slate-300">No routes yet</p>
              <p className="text-xs text-slate-500 mt-1">Create a route to begin.</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
              {roadmaps.map((roadmap) => (
                <button
                  key={roadmap.roadmap_id}
                  onClick={() => navigate(`/roadmap/${roadmap.roadmap_id}`)}
                  className="text-left bg-[#111322]/70 border border-white/5 rounded-2xl p-5 hover:border-blue-500/30 hover:bg-[#16192e] transition-all group"
                >
                  <div className="flex items-start justify-between gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                      <Route className="w-5 h-5 text-blue-400" />
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-blue-400 transition-colors" />
                  </div>
                  <h4 className="text-sm font-bold text-white line-clamp-2">
                    {roadmap.title}
                  </h4>
                  <p className="text-xs text-slate-400 mt-2 line-clamp-2 leading-relaxed">
                    {roadmap.description || roadmap.topic}
                  </p>
                  <div className="flex items-center justify-between mt-5 text-[10px] font-semibold text-slate-500">
                    <span>{roadmap.total_nodes || 0} nodes</span>
                    <span className="inline-flex items-center gap-1">
                      <Clock3 className="w-3 h-3" />
                      {formatDate(roadmap.created_at)}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </main>

      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="bg-[#111322] border border-white/10 text-white max-w-md">
          <DialogHeader>
            <DialogTitle className="text-sm font-bold uppercase tracking-wider flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-blue-400" />
              New Route
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4 pt-2">
            <Input
              placeholder="Topic, e.g. Agentic AI systems"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              className="bg-black/30 border-white/10 text-white placeholder:text-slate-600"
            />

            <Select value={level} onValueChange={setLevel}>
              <SelectTrigger className="bg-black/30 border-white/10 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="beginner">Beginner</SelectItem>
                <SelectItem value="intermediate">Intermediate</SelectItem>
                <SelectItem value="advanced">Advanced</SelectItem>
              </SelectContent>
            </Select>

            <Button
              onClick={handleGenerateRoadmap}
              disabled={isGenerating || !topic.trim()}
              className="w-full bg-gradient-to-r from-blue-600 to-violet-500 hover:opacity-90 text-white"
            >
              {isGenerating ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating...
                </span>
              ) : (
                "Generate Route"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

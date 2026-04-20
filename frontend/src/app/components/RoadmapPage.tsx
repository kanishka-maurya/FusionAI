import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
import { useRoadmap } from "../contexts/RoadmapContext";
import {
  ArrowLeft,
  GraduationCap,
  BookOpen,
  Target,
  Trophy,
  Brain,
  MessageCircle,
  CheckCircle,
  Clock,
  Star,
  TrendingUp,
  Sparkles,
} from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
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
};
export function RoadmapPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const {setRoadmap}=useRoadmap();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("beginner");
  const [isGenerating, setIsGenerating] = useState(false);

  const [roadmaps, setRoadmaps] = useState<Roadmap[]>([]);
  const [loadingRoadmaps, setLoadingRoadmaps] = useState(true);
  const fetchRoadmaps = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      const token = session?.access_token;
      const userId = session?.user?.id;
      const res = await fetch("http://localhost:8000/api/roadmap/user", {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });

      const data = await res.json();
      setRoadmaps(data.roadmaps || []);
    } catch (err) {
      console.error("Failed to fetch roadmaps:", err);
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
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      const token = session?.access_token;
      const userId = session?.user?.id;

      const response = await fetch(
        "http://localhost:8000/api/roadmap/generate",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic: topic,
            level: level,
            user_id: userId,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to generate roadmap");
      }

      const data = await response.json();
      await fetchRoadmaps();
      setRoadmap(data);
      navigate(`/roadmap/${data.roadmap_id}`);

      setIsCreateDialogOpen(false);
      setTopic("");
      setLevel("beginner");
    } catch (error) {
      console.error("Failed to generate roadmap:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/dashboard")}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  AI Roadmaps
                </h1>
                <p className="text-sm text-gray-600">
                  Your personalized learning paths
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <p className="text-sm">{user?.email}</p>
            <button onClick={logout}>Logout</button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-6 py-10">
        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-2xl font-bold">{roadmaps.length}</p>
            <p className="text-sm text-gray-600">Roadmaps</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-2xl font-bold">--</p>
            <p className="text-sm text-gray-600">Completed</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-2xl font-bold">--</p>
            <p className="text-sm text-gray-600">In Progress</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow">
            <p className="text-2xl font-bold">--</p>
            <p className="text-sm text-gray-600">Hours</p>
          </div>
        </div>

        {/* Roadmaps */}
        <div className="mb-8 flex justify-between">
          <h2 className="text-2xl font-bold">My Roadmaps</h2>
          <button
            onClick={() => setIsCreateDialogOpen(true)}
            className="bg-purple-600 text-white px-4 py-2 rounded-lg"
          >
            Create Roadmap
          </button>
        </div>

        {loadingRoadmaps ? (
          <p>Loading...</p>
        ) : roadmaps.length === 0 ? (
          <p>No roadmaps yet</p>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            {roadmaps.map((r) => (
              <div
                key={r.roadmap_id}
                onClick={() => navigate(`/roadmap/${r.roadmap_id}`)}
                className="bg-white p-6 rounded-xl border hover:shadow-lg cursor-pointer"
              >
                <h3 className="text-lg font-semibold">{r.title}</h3>
                <p className="text-sm text-gray-600 mt-1">
                  {r.description}
                </p>
                <div className="text-sm mt-3 text-gray-500">
                  {r.total_nodes} nodes
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Roadmap</DialogTitle>
          </DialogHeader>

          <Input
            placeholder="Topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />

          <Select value={level} onValueChange={setLevel}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="beginner">Beginner</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>

          <Button onClick={handleGenerateRoadmap} disabled={isGenerating}>
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
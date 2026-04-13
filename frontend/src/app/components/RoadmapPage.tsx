import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
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

export function RoadmapPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("beginner");
  const [isGenerating, setIsGenerating] = useState(false);

  const courses = [
    {
      id: "1",
      title: "Introduction to Machine Learning",
      level: "Beginner",
      progress: 45,
      lessons: 12,
      duration: "6 hours",
      category: "ML Fundamentals",
    },
    {
      id: "2",
      title: "Deep Learning with Neural Networks",
      level: "Intermediate",
      progress: 20,
      lessons: 18,
      duration: "10 hours",
      category: "Deep Learning",
    },
    {
      id: "3",
      title: "Natural Language Processing",
      level: "Advanced",
      progress: 0,
      lessons: 15,
      duration: "8 hours",
      category: "NLP",
    },
  ];

  const staffPicks = [
    "Understanding Transformers Architecture",
    "GPT Models Explained",
    "Computer Vision Basics",
    "Reinforcement Learning Intro",
  ];

  const recentQuizzes = [
    { title: "ML Basics Quiz", score: 85, date: "2 days ago" },
    { title: "Neural Networks Test", score: 92, date: "5 days ago" },
  ];

  const handleGenerateRoadmap = async () => {
    if (!topic.trim()) return;

    setIsGenerating(true);
    try {
      const {
            data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      const userId = session?.user?.id;
      
      console.log("generating");
      const response = await fetch("http://localhost:8000/api/roadmap/generate", {
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
      });

      if (!response.ok) {
        throw new Error("Failed to generate roadmap");
      }

      const data = await response.json();
      navigate(`/roadmap/${data.roadmap_id}`, {
        state: {
          topic: data.topic,
          level: level,
        },
      });

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
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  AI Tutor & Roadmap
                </h1>
                <p className="text-sm text-gray-600">
                  Your personalized learning journey
                </p>
              </div>
            </div>
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
              {user?.avatar && (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-9 h-9 rounded-full ring-2 ring-blue-100"
                />
              )}
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  {user?.name}
                </p>
                <p className="text-xs text-gray-500">{user?.email}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Learning Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <Target className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">3</p>
                <p className="text-sm text-gray-600">Courses Enrolled</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">24</p>
                <p className="text-sm text-gray-600">Lessons Completed</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <Trophy className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">88%</p>
                <p className="text-sm text-gray-600">Avg Quiz Score</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-orange-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">12h</p>
                <p className="text-sm text-gray-600">Learning Time</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content Column */}
          <div className="lg:col-span-2 space-y-8">
            {/* Courses Section */}
            <div>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">My Courses</h2>
                <button className="text-blue-600 hover:text-blue-700 font-medium text-sm">
                  View All
                </button>
              </div>

              <div className="space-y-4">
                {courses.map((course) => (
                  <div
                    key={course.id}
                    className="bg-white rounded-xl border border-gray-200 p-6 hover:border-blue-300 hover:shadow-lg transition-all cursor-pointer"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-lg font-semibold text-gray-900">
                            {course.title}
                          </h3>
                          <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
                            {course.level}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <span className="flex items-center gap-1">
                            <BookOpen className="w-4 h-4" />
                            {course.lessons} lessons
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            {course.duration}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Progress</span>
                        <span className="text-sm font-semibold text-blue-600">
                          {course.progress}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-600 to-blue-500 rounded-full transition-all"
                          style={{ width: `${course.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quizzes Section */}
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">
                Recent Quizzes
              </h2>
              <div className="space-y-4">
                {recentQuizzes.map((quiz, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl border border-gray-200 p-6 flex items-center justify-between hover:border-purple-300 hover:shadow-lg transition-all cursor-pointer"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                        <Brain className="w-6 h-6 text-purple-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900">
                          {quiz.title}
                        </h3>
                        <p className="text-sm text-gray-600">{quiz.date}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-2xl font-bold text-purple-600">
                        {quiz.score}%
                      </p>
                      <p className="text-xs text-gray-500">Score</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Create AI Roadmap */}
            <div className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 text-white">
              <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-2">Create AI Roadmap</h3>
              <p className="text-purple-100 text-sm mb-4">
                Generate a personalized learning path for any topic
              </p>
              <button
                onClick={() => setIsCreateDialogOpen(true)}
                className="w-full py-3 bg-white text-purple-700 rounded-lg font-medium hover:bg-purple-50 transition-colors"
              >
                Generate Roadmap
              </button>
            </div>

            {/* AI Tutor */}
            <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 text-white">
              <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center mb-4">
                <MessageCircle className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold mb-2">Ask AI Tutor</h3>
              <p className="text-blue-100 text-sm mb-4">
                Get instant help with your doubts and questions
              </p>
              <button className="w-full py-3 bg-white text-blue-700 rounded-lg font-medium hover:bg-blue-50 transition-colors">
                Start Chat
              </button>
            </div>

            {/* Staff Picks */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                <h3 className="font-bold text-gray-900">Staff Picks</h3>
              </div>
              <ul className="space-y-3">
                {staffPicks.map((pick, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-sm">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-600 mt-2 flex-shrink-0"></div>
                    <span className="text-gray-700 hover:text-blue-600 cursor-pointer">
                      {pick}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Learning Streak */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Trophy className="w-5 h-5 text-orange-500" />
                <h3 className="font-bold text-gray-900">Learning Streak</h3>
              </div>
              <div className="text-center py-4">
                <p className="text-4xl font-bold text-orange-600 mb-2">7</p>
                <p className="text-sm text-gray-600">Days in a row! 🔥</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Create Roadmap Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-600" />
              Create AI Roadmap
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="topic">Topic</Label>
              <Input
                id="topic"
                placeholder="e.g., Machine Learning, React, Python"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isGenerating) {
                    handleGenerateRoadmap();
                  }
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="level">Difficulty Level</Label>
              <Select value={level} onValueChange={setLevel}>
                <SelectTrigger id="level">
                  <SelectValue placeholder="Select level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => setIsCreateDialogOpen(false)}
              disabled={isGenerating}
            >
              Cancel
            </Button>
            <Button
              className="flex-1 bg-purple-600 hover:bg-purple-700"
              onClick={handleGenerateRoadmap}
              disabled={!topic.trim() || isGenerating}
            >
              {isGenerating ? (
                <>
                  <span className="animate-spin mr-2">⏳</span>
                  Generating...
                </>
              ) : (
                "Generate"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import {
  ArrowLeft,
  Newspaper,
  TrendingUp,
  Github,
  FileText,
  ExternalLink,
  Clock,
  Star,
  Eye,
  Calendar,
  Sparkles,
} from "lucide-react";

export function AINewsPage() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [selectedTab, setSelectedTab] = useState<"news" | "repos" | "papers">("news");

  const newsItems = [
    {
      id: "1",
      title: "OpenAI Releases GPT-5 with Revolutionary Reasoning Capabilities",
      source: "TechCrunch",
      date: "2 hours ago",
      category: "AI Models",
      views: "12.5K",
      summary:
        "OpenAI has unveiled GPT-5, featuring unprecedented reasoning abilities and multimodal understanding that surpasses previous benchmarks.",
    },
    {
      id: "2",
      title: "Google's Gemini Ultra 2.0 Achieves Human-Level Performance",
      source: "The Verge",
      date: "5 hours ago",
      category: "Research",
      views: "8.3K",
      summary:
        "Google's latest Gemini model demonstrates human-level performance across multiple benchmarks including reasoning, coding, and creative tasks.",
    },
    {
      id: "3",
      title: "Anthropic Announces Claude 5 with Enhanced Safety Features",
      source: "VentureBeat",
      date: "1 day ago",
      category: "AI Safety",
      views: "6.7K",
      summary:
        "Anthropic's Claude 5 introduces groundbreaking safety mechanisms and improved constitutional AI training methods.",
    },
  ];

  const trendingRepos = [
    {
      id: "1",
      name: "microsoft/autogen",
      description: "Enable Next-Gen Large Language Model Applications",
      stars: "25.3K",
      language: "Python",
      trend: "+2.1K this week",
    },
    {
      id: "2",
      name: "openai/gpt-5-api",
      description: "Official Python library for the GPT-5 API",
      stars: "18.7K",
      language: "Python",
      trend: "+5.2K this week",
    },
    {
      id: "3",
      name: "huggingface/transformers",
      description: "State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX",
      stars: "125K",
      language: "Python",
      trend: "+892 this week",
    },
  ];

  const researchPapers = [
    {
      id: "1",
      title: "Attention Is All You Need: A Retrospective Analysis",
      authors: "Vaswani et al.",
      date: "April 4, 2026",
      citations: "72,345",
      venue: "Nature Machine Intelligence",
    },
    {
      id: "2",
      title: "Constitutional AI: Harmlessness from AI Feedback",
      authors: "Bai et al.",
      date: "April 3, 2026",
      citations: "1,234",
      venue: "arXiv",
    },
    {
      id: "3",
      title: "Scaling Laws for Neural Language Models Revisited",
      authors: "Kaplan et al.",
      date: "April 2, 2026",
      citations: "3,567",
      venue: "ICML 2026",
    },
  ];

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
              <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-green-700 rounded-lg flex items-center justify-center">
                <Newspaper className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">AI News & Research</h1>
                <p className="text-sm text-gray-600">Stay updated with the latest AI developments</p>
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
                  className="w-9 h-9 rounded-full ring-2 ring-green-100"
                />
              )}
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{user?.name}</p>
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
        {/* AI Research Assistant */}
        <div className="bg-gradient-to-r from-green-600 to-green-700 rounded-2xl p-8 mb-12 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center">
              <Sparkles className="w-8 h-8" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">AI Research Assistant</h2>
              <p className="text-green-100">Get deep research summaries on today's top AI developments</p>
            </div>
          </div>
          <button className="mt-4 px-6 py-3 bg-white text-green-700 rounded-xl font-medium hover:bg-green-50 transition-colors">
            Start Deep Research
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-8 border-b border-gray-200">
          <button
            onClick={() => setSelectedTab("news")}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              selectedTab === "news"
                ? "border-green-600 text-green-600"
                : "border-transparent text-gray-600 hover:text-gray-900"
            }`}
          >
            <Newspaper className="w-5 h-5 inline mr-2" />
            Top News
          </button>
          <button
            onClick={() => setSelectedTab("repos")}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              selectedTab === "repos"
                ? "border-green-600 text-green-600"
                : "border-transparent text-gray-600 hover:text-gray-900"
            }`}
          >
            <Github className="w-5 h-5 inline mr-2" />
            Trending Repos
          </button>
          <button
            onClick={() => setSelectedTab("papers")}
            className={`px-6 py-3 font-medium border-b-2 transition-colors ${
              selectedTab === "papers"
                ? "border-green-600 text-green-600"
                : "border-transparent text-gray-600 hover:text-gray-900"
            }`}
          >
            <FileText className="w-5 h-5 inline mr-2" />
            Research Papers
          </button>
        </div>

        {/* Content */}
        {selectedTab === "news" && (
          <div className="space-y-6">
            {newsItems.map((news) => (
              <div
                key={news.id}
                className="bg-white rounded-xl border border-gray-200 p-6 hover:border-green-300 hover:shadow-lg transition-all cursor-pointer"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                        {news.category}
                      </span>
                      <span className="text-sm text-gray-500 flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {news.date}
                      </span>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2 hover:text-green-700">
                      {news.title}
                    </h3>
                    <p className="text-gray-600 mb-3">{news.summary}</p>
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>{news.source}</span>
                      <span className="flex items-center gap-1">
                        <Eye className="w-4 h-4" />
                        {news.views} views
                      </span>
                    </div>
                  </div>
                  <ExternalLink className="w-5 h-5 text-gray-400 ml-4 flex-shrink-0" />
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedTab === "repos" && (
          <div className="space-y-6">
            {trendingRepos.map((repo) => (
              <div
                key={repo.id}
                className="bg-white rounded-xl border border-gray-200 p-6 hover:border-green-300 hover:shadow-lg transition-all cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <Github className="w-5 h-5 text-gray-700" />
                      <h3 className="text-lg font-bold text-gray-900 hover:text-green-700">
                        {repo.name}
                      </h3>
                    </div>
                    <p className="text-gray-600 mb-4">{repo.description}</p>
                    <div className="flex items-center gap-6 text-sm text-gray-600">
                      <span className="flex items-center gap-1">
                        <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                        {repo.stars} stars
                      </span>
                      <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs">
                        {repo.language}
                      </span>
                      <span className="flex items-center gap-1 text-green-600">
                        <TrendingUp className="w-4 h-4" />
                        {repo.trend}
                      </span>
                    </div>
                  </div>
                  <ExternalLink className="w-5 h-5 text-gray-400 ml-4 flex-shrink-0" />
                </div>
              </div>
            ))}
          </div>
        )}

        {selectedTab === "papers" && (
          <div className="space-y-6">
            {researchPapers.map((paper) => (
              <div
                key={paper.id}
                className="bg-white rounded-xl border border-gray-200 p-6 hover:border-green-300 hover:shadow-lg transition-all cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <FileText className="w-5 h-5 text-gray-700" />
                      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium">
                        {paper.venue}
                      </span>
                    </div>
                    <h3 className="text-lg font-bold text-gray-900 mb-2 hover:text-green-700">
                      {paper.title}
                    </h3>
                    <p className="text-gray-600 mb-3">{paper.authors}</p>
                    <div className="flex items-center gap-6 text-sm text-gray-600">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        {paper.date}
                      </span>
                      <span>{paper.citations} citations</span>
                    </div>
                  </div>
                  <ExternalLink className="w-5 h-5 text-gray-400 ml-4 flex-shrink-0" />
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

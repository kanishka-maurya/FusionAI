import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { BookOpen, GraduationCap, Newspaper, Sparkles } from "lucide-react";

export function Dashboard() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const components = [
    {
      id: "notebook",
      title: "NotebookLM",
      description: "AI-powered research assistant with document analysis and chat interface",
      icon: BookOpen,
      color: "from-purple-500 to-purple-600",
      path: "/notebook",
      features: [
        "Upload and analyze documents",
        "Chat with your sources",
        "Create multiple notebook sessions",
        "Export conversations",
      ],
    },
    {
      id: "roadmap",
      title: "AI Tutor & Roadmap",
      description: "Personalized learning paths with courses, quizzes, and guidance",
      icon: GraduationCap,
      color: "from-blue-500 to-blue-600",
      path: "/roadmap",
      features: [
        "Personalized course recommendations",
        "Interactive quizzes",
        "Learning progress tracking",
        "AI tutor for doubts",
      ],
    },
    {
      id: "ai-news",
      title: "AI News & Research",
      description: "Daily curated AI news, research papers, and trending repositories",
      icon: Newspaper,
      color: "from-green-500 to-green-600",
      path: "/ai-news",
      features: [
        "Top AI news daily",
        "Trending GitHub repositories",
        "Latest research papers",
        "Deep research analysis",
      ],
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">AI Learning Platform</h1>
              <p className="text-sm text-gray-600">Your personalized AI assistant</p>
            </div>
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
              {user?.avatar && (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-9 h-9 rounded-full ring-2 ring-purple-100"
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
        {/* Welcome Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-3">
            Welcome back, {user?.name?.split(" ")[0]}! 👋
          </h2>
          <p className="text-lg text-gray-600">
            Choose a component to get started with your AI-powered learning journey
          </p>
        </div>

        {/* Component Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {components.map((component) => {
            const Icon = component.icon;
            return (
              <div
                key={component.id}
                onClick={() => navigate(component.path)}
                className="bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 cursor-pointer border border-gray-200 hover:border-gray-300 overflow-hidden group"
              >
                {/* Card Header */}
                <div className={`bg-gradient-to-r ${component.color} p-6`}>
                  <div className="w-14 h-14 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">
                    {component.title}
                  </h3>
                  <p className="text-white/90 text-sm leading-relaxed">
                    {component.description}
                  </p>
                </div>

                {/* Card Body */}
                <div className="p-6">
                  <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
                    Key Features
                  </h4>
                  <ul className="space-y-3">
                    {component.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <div className="w-5 h-5 rounded-full bg-gradient-to-r from-purple-100 to-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <div className="w-2 h-2 rounded-full bg-gradient-to-r from-purple-600 to-blue-600"></div>
                        </div>
                        <span className="text-sm text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Card Footer */}
                <div className="px-6 pb-6">
                  <div className="w-full py-3 px-4 bg-gradient-to-r from-gray-50 to-gray-100 text-gray-700 font-medium rounded-lg text-center group-hover:from-purple-600 group-hover:to-blue-600 group-hover:text-white transition-all">
                    Launch {component.title}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Quick Stats */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">0</p>
                <p className="text-sm text-gray-600">Notebooks Created</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <GraduationCap className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">0%</p>
                <p className="text-sm text-gray-600">Learning Progress</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl p-6 border border-gray-200">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <Newspaper className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">0</p>
                <p className="text-sm text-gray-600">Articles Read</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

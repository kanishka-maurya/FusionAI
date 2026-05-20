import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { BookOpen, GraduationCap, Newspaper } from "lucide-react";

export function Dashboard() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const components = [
    {
      id: "notebook",
      title: "Fusion Notebook",
      description: "AI-powered research assistant with document analysis and chat interface",
      icon: BookOpen,
      image: "https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcQfq2m4VKCgaIffEPZ75LziLRskUE0fUyfvH0RysF9V5WqhGuRJiEFsfAz_eutzhQgwhJqP1_uZLVrN-zM",
      color: "from-cyan-600/90 to-blue-600/90",
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
      title: "Pathfinder AI",
      description: "Personalized learning paths with courses, quizzes, and guidance",
      icon: GraduationCap,
      image: "https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcTZsliZ34xLCEyBUJponwdDVUArH5mty0bZWe2G1gYAf6nZZBnehWeR911025-84r8e-YKz-82hPkcIUXQ",
      color: "from-blue-600/90 to-purple-600/90",
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
      title: "Intelligence Feed",
      description: "Daily curated AI news, research papers, and trending repositories",
      icon: Newspaper,
      image: "https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcS-Z31YqNlrgE12mVixXHnoF-TCEhx2maBBhqbTLylfGZx7kWVbbzdf2Flrg92R0UJgQSHRctBiub-2vCE",
      color: "from-purple-600/90 to-pink-600/90",
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 flex flex-col">
      {/* Header */}
      <header className="bg-slate-900/40 backdrop-blur-md border-b border-white/10 shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Embedded FusionAI Logo Container */}
            <div
              className="inline-flex items-center justify-center w-10 h-10 rounded-xl shadow-md relative overflow-hidden flex-shrink-0"
              style={{
                background:
                  "linear-gradient(135deg, #0891b2 0%, #2563eb 40%, #7e22ce 100%)",
              }}
            >
              <div
                className="absolute inset-0 opacity-40 blur-md"
                style={{
                  background:
                    "radial-gradient(circle at center, #0891b2 0%, #2563eb 70%, #7e22ce 100%)",
                }}
              />
              <svg
                className="w-6 h-6 relative z-10"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <polygon
                  points="50,22 78,70 22,70"
                  fill="url(#dashPrismGradient)"
                  stroke="#7e22ce"
                  strokeWidth="2"
                  opacity="0.95"
                />
                <path d="M10 54 L32 58" stroke="#f1a598" strokeWidth="4" strokeLinecap="round" />
                <path d="M50 8 L50 30" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" />
                <path d="M90 54 L68 58" stroke="#a2c0e2" strokeWidth="4" strokeLinecap="round" />
                <circle cx="50" cy="50" r="4" fill="#ffffff" />
                <path d="M50 50 L36 62" stroke="#ffffff" strokeWidth="2" opacity="0.8" />
                <path d="M50 50 L64 62" stroke="#ffffff" strokeWidth="2" opacity="0.8" />
                <defs>
                  <linearGradient id="dashPrismGradient" x1="0" y1="0" x2="100" y2="100">
                    <stop offset="0%" stopColor="#f1a598" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#a2c0e2" stopOpacity="0.5" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-extrabold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent leading-none mb-0.5">
                FusionAI
              </h1>
              <p className="text-xs text-slate-400 font-medium">Your personalized AI assistant</p>
            </div>
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 shadow-sm">
              {user?.avatar && (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-8 h-8 rounded-full ring-2 ring-blue-500/30"
                />
              )}
              <div className="text-right hidden sm:block">
                <p className="text-xs font-semibold text-slate-200">{user?.name}</p>
                <p className="text-[10px] text-slate-400 font-medium leading-none">{user?.email}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-white/10 rounded-xl shadow-sm border border-white/10 transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Dashboard Wrapper */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-10">
        {/* Uniform Rectangular Transparent Container */}
        <div className="w-full bg-slate-900/40 border border-white/10 rounded-2xl shadow-2xl p-6 sm:p-10 backdrop-blur-md">
          
          {/* Welcome Section */}
          <div className="mb-10">
            <h2 className="text-3xl font-extrabold text-white tracking-tight mb-2">
              Welcome back, {user?.name?.split(" ")[0]}! 
            </h2>
            <p className="text-sm sm:text-base text-slate-400 font-medium">
              Choose a component to get started with your AI-powered learning journey
            </p>
          </div>

          {/* Component Cards Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {components.map((component) => {
              const Icon = component.icon;
              return (
                <div
                  key={component.id}
                  onClick={() => navigate(component.path)}
                  className="bg-slate-900/60 rounded-2xl shadow-sm hover:shadow-cyan-500/10 hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer border border-white/5 overflow-hidden group flex flex-col justify-between"
                >
                  <div>
                    {/* Card Image Banner Header */}
                    <div className="h-44 w-full relative overflow-hidden">
                      <img 
                        src={component.image} 
                        alt={component.title}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                      />
                      <div className={`absolute inset-0 bg-gradient-to-t ${component.color} opacity-80 mix-blend-multiply`} />
                      <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-transparent to-transparent" />
                      
                      <div className="absolute bottom-4 left-5 right-5 flex items-end justify-between">
                        <div>
                          <h3 className="text-xl font-bold text-white mb-0.5 tracking-wide drop-shadow-sm">
                            {component.title}
                          </h3>
                          <p className="text-white/80 text-xs font-medium max-w-[200px] line-clamp-1">
                            {component.description}
                          </p>
                        </div>
                        <div className="w-10 h-10 bg-white/10 backdrop-blur-md border border-white/20 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                      </div>
                    </div>

                    {/* Card Body */}
                    <div className="p-6">
                      <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-3">
                        Key Features
                      </h4>
                      <ul className="space-y-2.5">
                        {component.features.map((feature, idx) => (
                          <li key={idx} className="flex items-start gap-2.5">
                            <div className="w-4 h-4 rounded-full bg-white/5 border border-white/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <div className="w-1 h-1 rounded-full bg-gradient-to-r from-cyan-400 to-blue-400" />
                            </div>
                            <span className="text-xs font-medium text-slate-300 group-hover:text-white transition-colors leading-normal">
                              {feature}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Card Footer */}
                  <div className="px-6 pb-6">
                    <div className="w-full py-2.5 px-4 bg-white/5 font-semibold rounded-xl text-xs text-slate-300 text-center border border-white/5 group-hover:bg-gradient-to-r group-hover:from-cyan-500 group-hover:to-blue-600 group-hover:text-white group-hover:border-transparent group-hover:shadow-md transition-all duration-300">
                      Launch Hub
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Quick Stats Grid */}
          <div className="mt-10 pt-10 border-t border-white/10 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-900/40 rounded-2xl p-5 border border-white/5 shadow-sm hover:border-cyan-500/30 transition-colors flex items-center gap-4">
              <div className="w-11 h-11 bg-cyan-500/10 rounded-xl flex items-center justify-center flex-shrink-0">
                <BookOpen className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <p className="text-xl font-black text-white">0</p>
                <p className="text-xs font-semibold text-slate-400">Notebooks Syncing</p>
              </div>
            </div>
            <div className="bg-slate-900/40 rounded-2xl p-5 border border-white/5 shadow-sm hover:border-blue-500/30 transition-colors flex items-center gap-4">
              <div className="w-11 h-11 bg-blue-500/10 rounded-xl flex items-center justify-center flex-shrink-0">
                <GraduationCap className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <p className="text-xl font-black text-white">0%</p>
                <p className="text-xs font-semibold text-slate-400">Learning Milestone</p>
              </div>
            </div>
            <div className="bg-slate-900/40 rounded-2xl p-5 border border-white/5 shadow-sm hover:border-purple-500/30 transition-colors flex items-center gap-4">
              <div className="w-11 h-11 bg-purple-500/10 rounded-xl flex items-center justify-center flex-shrink-0">
                <Newspaper className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-xl font-black text-white">0</p>
                <p className="text-xs font-semibold text-slate-400">Articles Cataloged</p>
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
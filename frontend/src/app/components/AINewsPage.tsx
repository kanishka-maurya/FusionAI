import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
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
  Cpu,
  Terminal,
  Send,
  Loader2,
  CheckCircle2
} from "lucide-react";

// Strictly typed model interface mapping exactly to FusionAI background workers
interface IngestedItem {
  title: string;
  url: string;
  source: "github" | "papers" | "news";
  created_at: string;
  fetched_at: string;
  content: string; // Extracted full document payload
  meta: {
    stars?: number;
    language?: string | null;
    source_name?: string;
  };
}

export function AINewsPage() {
  const [items, setItems] = useState<IngestedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [selectedTab, setSelectedTab] = useState<"news" | "repos" | "papers">("news");

  // Pipeline Query States
  const [queryInput, setQueryInput] = useState("");
  const [queryStatus, setQueryStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");
  const [statusMessage, setStatusMessage] = useState("");

  const fetchData = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      
      const token = session?.access_token;
      // Points exactly to your live FusionAI ingestion cluster base path
      const res = await fetch("http://localhost:8000/ai-news/get", {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      }); 
      const json = await res.json();
      
      if (json.data) {
        setItems(json.data);
      }
      if (json.last_updated) {
        setLastUpdated(json.last_updated);
      }
    } catch (err) {
      console.error("Telemetry matrix acquisition fault:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Ingestion cycle interval matched seamlessly against the background worker cadence
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Submit query directly to your Redis QUERY_QUEUE broker via FastAPI
  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) return;

    setQueryStatus("submitting");
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      
      const token = session?.access_token;
      const res = await fetch("http://localhost:8000/ai-news/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query: queryInput }),
      });
      
      const result = await res.json();
      
      if (result.success) {
        setQueryStatus("success");
        setStatusMessage("Vector query pushed to Redis queue securely.");
        setQueryInput("");
        // Reset status message back to normal layout after a short delay
        setTimeout(() => setQueryStatus("idle"), 4000);
      } else {
        setQueryStatus("error");
        setStatusMessage(result.message || "Queue insertion failure.");
      }
    } catch (err) {
      console.error("Query pipeline injection fault:", err);
      setQueryStatus("error");
      setStatusMessage("Could not connect to the pipeline controller cluster.");
    }
  };

  // Isolate telemetry layers cleanly based on the 'source' identity provided by your FastAPI routers
  const newsItems = items.filter((item) => item.source === "news");
  const trendingRepos = items.filter((item) => item.source === "github");
  const researchPapers = items.filter((item) => item.source === "papers");

  // Helper utility to parse timestamps cleanly without breaking formatting rules
  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="min-h-screen bg-[#0d0e1b] text-slate-100 font-sans antialiased">
      {/* STICKY BLUR HEADER PANEL */}
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
              <div className="w-10 h-10 bg-gradient-to-tr from-cyan-600 to-blue-500 rounded-xl flex items-center justify-center border border-cyan-400/20 shadow-lg shadow-cyan-500/10">
                <Newspaper className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-white uppercase tracking-wider">Intelligence Vector Feed</h1>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Real-time ingestion tracking of global AI telemetry
                  {lastUpdated && ` • Sync: ${formatTime(lastUpdated)}`}
                </p>
              </div>
            </div>
          </div>

          {/* USER CONTEXT AVATAR */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-3 py-1.5 bg-white/[0.01] border border-white/5 rounded-xl">
              {user?.avatar && (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-7 h-7 rounded-full ring-2 ring-cyan-500/20 object-cover"
                />
              )}
              <div className="text-right hidden sm:block">
                <p className="text-xs font-bold text-slate-200">{user?.name || "Operator"}</p>
                <p className="text-[10px] text-slate-500 font-mono">{user?.email || "active-session"}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="px-3 py-1.5 text-xs font-bold text-slate-400 hover:text-rose-400 bg-white/[0.01] hover:bg-rose-500/5 border border-white/5 hover:border-rose-500/20 rounded-xl transition-all"
            >
              Disconnect
            </button>
          </div>
        </div>
      </header>

      {/* CORE WORKSPACE CONTEXT */}
      <main className="max-w-7xl mx-auto px-6 py-10 space-y-10">
        
        {/* INTERACTIVE PIPELINE ACTION CARD - MAPS DIRECTLY TO @router.post('/query') */}
        <div className="relative bg-[#111322] border border-white/5 rounded-2xl p-8 overflow-hidden shadow-xl shadow-black/20">
          <div className="absolute top-0 right-0 p-8 opacity-[0.03] pointer-events-none">
            <Cpu className="w-40 h-40 text-cyan-400" />
          </div>
          <div className="relative space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-cyan-500/10 border border-cyan-500/20 rounded-xl flex items-center justify-center shrink-0">
                <Terminal className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-md font-bold text-white tracking-tight">FusionAI Live Core Broker</h2>
                <p className="text-xs text-slate-400 mt-1 max-w-xl leading-relaxed">
                  Investigate and Analyze any topic deeply !
                </p>
              </div>
            </div>

            {/* INTERACTIVE INPUT BAR */}
            <form onSubmit={handleQuerySubmit} className="max-w-3xl flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  disabled={queryStatus === "submitting"}
                  placeholder="Enter custom telemetry evaluation prompt..."
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20 transition-all font-mono"
                />
              </div>
              <button
                type="submit"
                disabled={queryStatus === "submitting" || !queryInput.trim()}
                className="px-5 py-3 bg-gradient-to-r from-cyan-600 to-blue-500 text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 active:scale-[0.98] transition-all shadow-md shadow-cyan-500/10 flex items-center justify-center gap-2 disabled:opacity-40 disabled:scale-100 min-w-[140px]"
              >
                {queryStatus === "submitting" ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    QUEUING...
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    DISPATCH
                  </>
                )}
              </button>
            </form>

            {/* REALTIME SYSTEM RESPONSE FEEDBACK */}
            {queryStatus !== "idle" && (
              <div className={`flex items-center gap-2 text-xs font-mono p-3 rounded-lg border max-w-3xl ${
                queryStatus === "success" 
                  ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400" 
                  : queryStatus === "error"
                  ? "bg-rose-500/5 border-rose-500/20 text-rose-400"
                  : "bg-white/[0.02] border-white/5 text-slate-400"
              }`}>
                {queryStatus === "success" && <CheckCircle2 className="w-4 h-4 shrink-0" />}
                {queryStatus === "submitting" && <Loader2 className="w-4 h-4 animate-spin shrink-0" />}
                <span>{statusMessage || "Syncing pipeline buffer index..."}</span>
              </div>
            )}
          </div>
        </div>

        {/* CONTROLLER NAVIGATION TABS */}
        <div className="flex gap-2 border-b border-white/5 font-mono text-xs">
          <button
            onClick={() => setSelectedTab("news")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "news"
                ? "border-cyan-500 text-cyan-400 bg-cyan-500/[0.02]"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Newspaper className="w-4 h-4" />
            Top Bulletins ({newsItems.length})
          </button>
          <button
            onClick={() => setSelectedTab("repos")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "repos"
                ? "border-cyan-500 text-cyan-400 bg-cyan-500/[0.02]"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Github className="w-4 h-4" />
            Active Code Repos ({trendingRepos.length})
          </button>
          <button
            onClick={() => setSelectedTab("papers")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "papers"
                ? "border-cyan-500 text-cyan-400 bg-cyan-500/[0.02]"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <FileText className="w-4 h-4" />
            Research Literature ({researchPapers.length})
          </button>
        </div>

        {/* FEED GRID CONTENT INTERFACE */}
        <div className="space-y-4">
          {loading ? (
            <div className="text-center py-20 font-mono text-xs text-slate-500 tracking-widest animate-pulse">
              INGESTING DATA ARTIFACTS FROM HOST INTERFACE...
            </div>
          ) : items.length === 0 ? (
            <div className="text-center py-20 font-mono text-xs text-slate-500 border border-dashed border-white/5 rounded-xl">
              No vectors recorded within the sliding 7-day memory horizon.
            </div>
          ) : null}

          {/* TOP NEWS FEED */}
          {!loading && selectedTab === "news" &&
            newsItems.map((news, idx) => (
              <a
                href={news.url}
                target="_blank"
                rel="noreferrer"
                key={idx}
                className="group bg-white/[0.01] border border-white/5 rounded-xl p-5 hover:bg-white/[0.02] hover:border-cyan-500/20 transition-all duration-200 flex items-start justify-between gap-6 block"
              >
                <div className="space-y-3 min-w-0 flex-1">
                  <div className="flex items-center gap-3 text-[10px] font-mono">
                    <span className="px-2 py-0.5 bg-cyan-500/10 text-cyan-400 rounded border border-cyan-500/20 font-bold uppercase tracking-wide">
                      {news.meta.source_name || "Global Media"}
                    </span>
                    <span className="text-slate-500 flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      {formatTime(news.created_at)}
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-slate-200 group-hover:text-cyan-400 transition-colors">
                    {news.title}
                  </h3>
                  {news.content && (
                    <p className="text-xs text-slate-400 leading-relaxed max-w-4xl line-clamp-2">
                      {news.content}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
                    <span className="text-slate-400 font-semibold">Source Type: Ingestion Router</span>
                    <span className="text-slate-600">•</span>
                    <span className="flex items-center gap-1">
                      <Eye className="w-3.5 h-3.5" />
                      Tracked at {formatTime(news.fetched_at)}
                    </span>
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 shrink-0 transition-colors mt-1" />
              </a>
            ))}

          {/* CODE REPOSITORIES FEED */}
          {!loading && selectedTab === "repos" &&
            trendingRepos.map((repo, idx) => (
              <a
                href={repo.url}
                target="_blank"
                rel="noreferrer"
                key={idx}
                className="group bg-white/[0.01] border border-white/5 rounded-xl p-5 hover:bg-white/[0.02] hover:border-cyan-500/20 transition-all duration-200 flex items-start justify-between gap-6 block"
              >
                <div className="space-y-2.5 min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <Github className="w-4 h-4 text-slate-400 group-hover:text-cyan-400 transition-colors" />
                    <h3 className="text-sm font-bold text-slate-200 group-hover:text-cyan-400 transition-colors font-mono">
                      {repo.title}
                    </h3>
                  </div>
                  {repo.content && (
                    <p className="text-xs text-slate-400 leading-relaxed max-w-4xl line-clamp-2">
                      {repo.content}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
                    <span className="flex items-center gap-1 text-amber-400/80">
                      <Star className="w-3.5 h-3.5 fill-amber-400/20 text-amber-400" />
                      {repo.meta.stars?.toLocaleString() || "0"} Stars
                    </span>
                    {repo.meta.language && (
                      <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded border border-blue-500/20 text-[9px] font-bold uppercase">
                        {repo.meta.language}
                      </span>
                    )}
                    <span className="flex items-center gap-1 text-cyan-400 font-bold">
                      <TrendingUp className="w-3.5 h-3.5" />
                      Created {formatTime(repo.created_at)}
                    </span>
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 shrink-0 transition-colors mt-0.5" />
              </a>
            ))}

          {/* RESEARCH LITERATURE FEED */}
          {!loading && selectedTab === "papers" &&
            researchPapers.map((paper, idx) => (
              <a
                href={paper.url}
                target="_blank"
                rel="noreferrer"
                key={idx}
                className="group bg-white/[0.01] border border-white/5 rounded-xl p-5 hover:bg-white/[0.02] hover:border-cyan-500/20 transition-all duration-200 flex items-start justify-between gap-6 block"
              >
                <div className="space-y-3 min-w-0 flex-1">
                  <div className="flex items-center gap-2 text-[10px] font-mono">
                    <div className="w-5 h-5 rounded bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                      <FileText className="w-3 h-3 text-purple-400" />
                    </div>
                    <span className="px-2 py-0.5 bg-purple-500/10 text-purple-400 rounded border border-purple-500/20 text-[9px] font-bold uppercase">
                      arXiv:cs.AI
                    </span>
                  </div>
                  <h3 className="text-sm font-bold text-slate-200 group-hover:text-cyan-400 transition-colors">
                    {paper.title}
                  </h3>
                  {paper.content && (
                    <p className="text-xs text-slate-400 leading-relaxed line-clamp-3 bg-white/[0.01] p-3 rounded-lg border border-white/5 mt-2">
                      {paper.content}
                    </p>
                  )}
                  <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5" />
                      Published: {formatTime(paper.created_at)}
                    </span>
                    <span className="text-slate-600">•</span>
                    <span className="text-slate-400">Ingested via Core Extractor Pipeline</span>
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 shrink-0 transition-colors mt-1" />
              </a>
            ))}
        </div>
      </main>
    </div>
  );
}
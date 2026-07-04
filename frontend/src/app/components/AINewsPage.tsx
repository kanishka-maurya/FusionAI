import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
import {
  ArrowLeft,
  Newspaper,
  Github,
  FileText,
  ExternalLink,
  Clock,
  Star,
  Calendar,
  Terminal,
  Send,
  Loader2,
  CheckCircle2,
  Eye,
  Search,
  Network,
  Sparkles,
} from "lucide-react";

interface IngestedItem {
  title: string;
  url: string;
  source: "github" | "papers" | "news";
  created_at: string;
  fetched_at: string;
  content?: string;
  meta?: {
    stars?: number;
    language?: string | null;
    source_name?: string;
  };
}

export function AINewsPage() {
  const [items, setItems] = useState<IngestedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [queryInput, setQueryInput] = useState("");
  const [queryStatus, setQueryStatus] = useState<
    "idle" | "submitting" | "success" | "error"
  >("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [queryResponse, setQueryResponse] = useState<any>(null);
  const [selectedTab, setSelectedTab] = useState<"news" | "repos" | "papers">(
    "news"
  );

  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const formatTime = (isoString: string) => {
    if (!isoString) return "";
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoString;
    }
  };

  const fetchData = async () => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;

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
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) {
      return;
    }

    setQueryStatus("submitting");
    setStatusMessage("");
    setQueryResponse(null);

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
        body: JSON.stringify({
          query: queryInput,
        }),
      });

      const result = await res.json();

      if (result.success) {
        const userView = result.response?.user_view;

        if (!userView) {
          throw new Error("No user-facing analysis returned.");
        }

        setQueryStatus("success");
        setStatusMessage("FusionAI recursive analysis completed.");
        setQueryResponse(userView);
      } else {
        setQueryStatus("error");
        setStatusMessage(result.message || "Pipeline execution failed.");
      }
    } catch (err) {
      console.error("Query error:", err);
      setQueryStatus("error");
      setStatusMessage("Unable to connect to backend.");
    }
  };

  const newsItems = items.filter((item) => item.source === "news");
  const trendingRepos = items.filter((item) => item.source === "github");
  const researchPapers = items.filter((item) => item.source === "papers");
  const analysis = queryResponse;
  const similarTopics = analysis?.similar_topics ?? [];
  const followUpQuestions =
    analysis?.follow_up_questions ?? [];
  const generatedAnswers =
    analysis?.answers ?? [];
  const evidence = analysis?.retrieved_evidence ?? [];
  const topicName =
    analysis?.topic_name ??
    analysis?.topic?.topic ??
    "No dominant topic found";

  return (
    <div className="min-h-screen bg-[#0d0e1b] text-slate-100 font-sans antialiased">
      {/* HEADER */}
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
                <h1 className="text-sm font-bold text-white uppercase tracking-wider">
                  Intelligence Vector Feed
                </h1>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Recursive Explainable Graph-RAG Intelligence
                  {lastUpdated && ` • Sync: ${formatTime(lastUpdated)}`}
                </p>
              </div>
            </div>
          </div>

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
                <p className="text-xs font-bold text-slate-200">
                  {user?.name || "Operator"}
                </p>
                <p className="text-[10px] text-slate-500 font-mono">
                  {user?.email}
                </p>
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

      {/* MAIN */}
      <main className="max-w-7xl mx-auto px-6 py-10 space-y-10">
        {/* QUERY PANEL */}
        <div className="bg-[#111322] border border-white/5 rounded-2xl p-8">
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-cyan-500/10 border border-cyan-500/20 rounded-xl flex items-center justify-center">
                <Terminal className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-md font-bold text-white tracking-tight">
                  FusionAI Recursive Intelligence Engine
                </h2>
                <p className="text-xs text-slate-400 mt-1 max-w-xl leading-relaxed">
                  Deep explainable recursive retrieval intelligence.
                </p>
              </div>
            </div>

            <form
              onSubmit={handleQuerySubmit}
              className="max-w-4xl flex flex-col sm:flex-row gap-3"
            >
              <div className="relative flex-1">
                <input
                  type="text"
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  disabled={queryStatus === "submitting"}
                  placeholder="Enter intelligence query..."
                  className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 font-mono"
                />
              </div>

              <button
                type="submit"
                disabled={queryStatus === "submitting" || !queryInput.trim()}
                className="px-5 py-3 bg-gradient-to-r from-cyan-600 to-blue-500 text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 transition-all flex items-center justify-center gap-2 disabled:opacity-40 min-w-[140px]"
              >
                {queryStatus === "submitting" ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    PROCESSING...
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    DISPATCH
                  </>
                )}
              </button>
            </form>

            {/* STATUS */}
            {queryStatus !== "idle" && (
              <div
                className={`flex items-center gap-2 text-xs font-mono p-3 rounded-lg border max-w-3xl ${
                  queryStatus === "success"
                    ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-400"
                    : queryStatus === "error"
                    ? "bg-rose-500/5 border-rose-500/20 text-rose-400"
                    : "bg-white/[0.02] border-white/5 text-slate-400"
                }`}
              >
                {queryStatus === "success" && (
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                )}
                {queryStatus === "submitting" && (
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                )}
                <span>{statusMessage}</span>
              </div>
            )}

            {/* FINAL OUTPUT */}
            {queryResponse && (
              <div className="mt-6 space-y-8">
                <div className="grid grid-cols-1 lg:grid-cols-[1.35fr_0.9fr] gap-5">
                  <div className="bg-black/30 border border-cyan-500/20 rounded-2xl p-6 space-y-5">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
                        <Eye className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider">
                          Research Insight
                        </h3>
                        <p className="text-xs text-slate-500 font-mono">
                          Dominant topic: {topicName}
                        </p>
                      </div>
                    </div>

                    <div className="bg-[#0d111c] border border-white/5 rounded-xl p-5">
                      <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {analysis?.summary || "FusionAI retrieved context for this query, but no final narrative answer was generated."}
                      </p>
                    </div>

                    {generatedAnswers.length > 0 && (
                      <div className="space-y-3">
                        {generatedAnswers.map((item: any, idx: number) => (
                          <div key={idx} className="bg-[#0d111c] border border-white/5 rounded-xl p-4">
                            <p className="text-xs text-cyan-400 font-bold uppercase mb-2">
                              {item.question || `Generated Answer ${idx + 1}`}
                            </p>
                            <p className="text-xs text-slate-300 leading-relaxed">
                              {item.answer}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="bg-black/30 border border-white/5 rounded-2xl p-6 space-y-5">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                        <Network className="w-5 h-5 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-blue-400 uppercase tracking-wider">
                          You can also search for
                        </h3>
                        <p className="text-xs text-slate-500 font-mono">
                          Related graph topics
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2">
                      {similarTopics.length === 0 ? (
                        <p className="text-xs text-slate-500">No related topics were found for this query.</p>
                      ) : (
                        similarTopics.map((item: any, idx: number) => {
                          const topic = item.topic || item;
                          return (
                            <button
                              key={`${topic}-${idx}`}
                              type="button"
                              onClick={() => setQueryInput(topic)}
                              className="px-3 py-2 rounded-lg bg-white/[0.03] hover:bg-cyan-500/10 border border-white/10 hover:border-cyan-500/30 text-xs text-slate-300 hover:text-cyan-300 transition-all flex items-center gap-2"
                              title={item.reason || "Search this related topic"}
                            >
                              <Search className="w-3.5 h-3.5" />
                              {topic}
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>

                {(followUpQuestions.length > 0 || evidence.length > 0) && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    {followUpQuestions.length > 0 && (
                      <div className="bg-black/30 border border-white/5 rounded-2xl p-6">
                        <h4 className="text-sm font-bold text-cyan-400 uppercase tracking-wide mb-5 flex items-center gap-2">
                          <Sparkles className="w-4 h-4" />
                          Follow-up angles
                        </h4>
                        <div className="space-y-3">
                          {followUpQuestions.map((item: any, idx: number) => (
                            <div key={idx} className="bg-[#0d111c] border border-white/5 rounded-xl p-4">
                              <p className="text-sm font-semibold text-white leading-relaxed">
                                {item.question}
                              </p>
                              {item.knowledge_gap && (
                                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                                  {item.knowledge_gap}
                                </p>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {evidence.length > 0 && (
                      <div className="bg-black/30 border border-white/5 rounded-2xl p-6">
                        <h4 className="text-sm font-bold text-cyan-400 uppercase tracking-wide mb-5">
                          Retrieved Evidence
                        </h4>
                        <div className="space-y-3">
                          {evidence.slice(0, 4).map((item: any, idx: number) => (
                            <div key={idx} className="bg-[#0d111c] border border-white/5 rounded-xl p-4">
                              <p className="text-xs text-slate-300 leading-relaxed line-clamp-4">
                                {item.summary || "No summary available."}
                              </p>
                              {item.key_points?.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-3">
                                  {item.key_points.slice(0, 3).map((kp: any, kpIdx: number) => (
                                    <span key={kpIdx} className="px-2 py-1 rounded-md bg-cyan-500/5 border border-cyan-500/10 text-[10px] text-cyan-300">
                                      {typeof kp === "string" ? kp : kp.text}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* TABS */}
        <div className="flex gap-2 border-b border-white/5 font-mono text-xs">
          <button
            onClick={() => setSelectedTab("news")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "news"
                ? "border-cyan-500 text-cyan-400"
                : "border-transparent text-slate-400"
            }`}
          >
            <Newspaper className="w-4 h-4" />
            News ({newsItems.length})
          </button>

          <button
            onClick={() => setSelectedTab("repos")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "repos"
                ? "border-cyan-500 text-cyan-400"
                : "border-transparent text-slate-400"
            }`}
          >
            <Github className="w-4 h-4" />
            Repositories ({trendingRepos.length})
          </button>

          <button
            onClick={() => setSelectedTab("papers")}
            className={`px-4 py-3 font-bold border-b-2 transition-all flex items-center gap-2 tracking-wide uppercase ${
              selectedTab === "papers"
                ? "border-cyan-500 text-cyan-400"
                : "border-transparent text-slate-400"
            }`}
          >
            <FileText className="w-4 h-4" />
            Papers ({researchPapers.length})
          </button>
        </div>

        {/* FEED DETAILS LIST CONTAINER */}
        <div className="mt-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20 gap-3 text-slate-400">
              <Loader2 className="w-8 h-8 animate-spin text-cyan-500" />
              <p className="text-xs font-mono">Syncing system matrices...</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {/* NEWS TAB DISPLAY */}
              {selectedTab === "news" &&
                (newsItems.length === 0 ? (
                  <p className="text-xs font-mono text-slate-500 py-8 text-center">No raw news streams connected.</p>
                ) : (
                  newsItems.map((item, idx) => (
                    <div key={idx} className="bg-[#111322] border border-white/5 rounded-xl p-5 hover:border-white/10 transition-all flex flex-col justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-[10px] font-mono text-cyan-400">
                          <span className="px-2 py-0.5 bg-cyan-500/10 border border-cyan-500/20 rounded-md uppercase tracking-wider">
                            {item.meta?.source_name || "Global Stream"}
                          </span>
                          <span className="flex items-center gap-1 text-slate-500">
                            <Clock className="w-3 h-3" /> {formatTime(item.created_at)}
                          </span>
                        </div>
                        <h3 className="text-sm font-semibold text-white tracking-tight hover:text-cyan-400 transition-colors">
                          <a href={item.url} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 group">
                            {item.title}
                            <ExternalLink className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                          </a>
                        </h3>
                        {item.content && (
                          <p className="text-xs text-slate-400 leading-relaxed line-clamp-3">
                            {item.content}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                ))}

              {/* REPOSITORIES TAB DISPLAY */}
              {selectedTab === "repos" &&
                (trendingRepos.length === 0 ? (
                  <p className="text-xs font-mono text-slate-500 py-8 text-center">No indexed repository logs available.</p>
                ) : (
                  trendingRepos.map((item, idx) => (
                    <div key={idx} className="bg-[#111322] border border-white/5 rounded-xl p-5 hover:border-white/10 transition-all flex flex-col justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3 text-[10px] font-mono">
                            {item.meta?.language && (
                              <span className="px-2 py-0.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 rounded-md tracking-wide">
                                {item.meta.language}
                              </span>
                            )}
                            <span className="flex items-center gap-1 text-amber-400 font-bold">
                              <Star className="w-3 h-3 fill-amber-400/20" /> {item.meta?.stars?.toLocaleString() || 0}
                            </span>
                          </div>
                          <span className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
                            <Clock className="w-3 h-3" /> {formatTime(item.created_at)}
                          </span>
                        </div>
                        <h3 className="text-sm font-semibold text-white tracking-tight hover:text-cyan-400 transition-colors">
                          <a href={item.url} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 group">
                            {item.title}
                            <ExternalLink className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                          </a>
                        </h3>
                        {item.content && (
                          <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">
                            {item.content}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                ))}

              {/* PAPERS TAB DISPLAY */}
              {selectedTab === "papers" &&
                (researchPapers.length === 0 ? (
                  <p className="text-xs font-mono text-slate-500 py-8 text-center">No empirical research records captured.</p>
                ) : (
                  researchPapers.map((item, idx) => (
                    <div key={idx} className="bg-[#111322] border border-white/5 rounded-xl p-5 hover:border-white/10 transition-all flex flex-col justify-between gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-[10px] font-mono text-purple-400">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" /> Published: {formatTime(item.created_at)}
                          </span>
                        </div>
                        <h3 className="text-sm font-semibold text-white tracking-tight hover:text-cyan-400 transition-colors">
                          <a href={item.url} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 group">
                            {item.title}
                            <ExternalLink className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                          </a>
                        </h3>
                        {item.content && (
                          <p className="text-xs text-slate-400 leading-relaxed border-l border-white/5 pl-3 italic">
                            {item.content}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

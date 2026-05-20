import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
import { useNotebook } from "../contexts/NotebookContext";
import {
  ArrowLeft,
  Plus,
  BookOpen,
  Clock,
  FileText,
  Sparkles,
} from "lucide-react";

interface Notebook {
  id: string;
  name: string;
  description: string;
  sourcesCount: number;
  messagesCount: number;
  lastModified: string;
  createdAt: string;
}

export function NotebookSessions() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { setNotebook } = useNotebook();
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newNotebookName, setNewNotebookName] = useState("");
  const [newNotebookDesc, setNewNotebookDesc] = useState("");

  useEffect(() => {
    fetchNotebooks();
  }, []);

  const handleCreateNotebook = async () => {
    if (!newNotebookName.trim()) return;

    const {
      data: { session },
    } = await supabase.auth.getSession();

    const userId = session?.user?.id;

    const { data, error } = await supabase
      .from("notebooks")
      .insert([
        {
          user_id: userId,
          name: newNotebookName,
          description: newNotebookDesc,
        },
      ])
      .select()
      .single();

    if (error) {
      console.error(error);
      alert("Failed to create notebook");
      return;
    }

    const newNotebookId = data.notebook_id;

    const newNotebook: Notebook = {
      id: newNotebookId,
      name: data.name,
      description: data.description,
      sourcesCount: 0,
      messagesCount: 0,
      lastModified: "Just now",
      createdAt: new Date().toLocaleDateString("en-US", {
        month: "long",
        day: "numeric",
        year: "numeric",
      }),
    };

    setNotebook({
      notebook_id: newNotebookId,
      name: data.name,
    });

    setNotebooks((prev) => [newNotebook, ...prev]);

    setNewNotebookName("");
    setNewNotebookDesc("");
    setShowCreateModal(false);

    navigate(`/notebook/${newNotebookId}`);
  };

  const fetchNotebooks = async () => {
    const { data, error } = await supabase
      .from("notebooks")
      .select("notebook_id, name, description, created_at")
      .order("created_at", { ascending: false });

    if (error) {
      console.error(error);
      return;
    }

    const formatted = data.map((n) => ({
      id: n.notebook_id,
      name: n.name,
      description: n.description,
      sourcesCount: 0,
      messagesCount: 0,
      lastModified: "Recently",
      createdAt: new Date(n.created_at).toLocaleDateString(),
    }));

    setNotebooks(formatted);
  };

  return (
    <div
      className="min-h-screen font-sans antialiased relative text-[#94a3b8] selection:bg-blue-500/50 selection:text-white
    before:absolute before:inset-0 before:-z-10 
    before:bg-[url('https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcQfq2m4VKCgaIffEPZ75LziLRskUE0fUyfvH0RysF9V5WqhGuRJiEFsfAz_eutzhQgwhJqP1_uZLVrN-zM')] 
    before:bg-cover before:bg-center before:bg-no-repeat before:bg-fixed 
    before:blur-md before:brightness-[0.8]"
    >
      {/* Dashboard Top Header Bar */}
      <header className="bg-[#111322]/60 border-b border-white/5 sticky top-0 z-40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/dashboard")}
              className="p-2 hover:bg-white/5 rounded-xl text-slate-400 hover:text-white transition-all"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-gradient-to-tr from-blue-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
                <BookOpen className="w-4 h-4 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-md font-bold tracking-tight text-white leading-none">
                    FusionAI
                  </h1>
                  <span className="text-[10px] bg-blue-500/10 text-blue-400 px-1.5 py-0.5 rounded font-medium border border-blue-500/20">
                    Notebook
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">
                  Your personalized learning journey
                </p>
              </div>
            </div>
          </div>

          {/* User Section (Matched with your exact top-right corner layout) */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-3 py-1.5 rounded-xl bg-[#111322] border border-white/5">
              <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-600 flex items-center justify-center text-white text-xs font-bold overflow-hidden ring-1 ring-white/10">
                {user?.avatar ? (
                  <img
                    src={user.avatar}
                    alt={user.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span>{user?.name?.charAt(0) || "S"}</span>
                )}
              </div>
              <div className="text-right hidden md:block">
                <p className="text-xs font-semibold text-white leading-tight">
                  {user?.name || "Shourya Mishra"}
                </p>
                <p className="text-[10px] text-slate-400 font-normal">
                  {user?.email || "shouryamishra55@gmail.com"}
                </p>
              </div>
            </div>
            <button
              onClick={logout}
              className="px-4 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-[#111322] hover:bg-[#16192e] rounded-xl transition-all border border-white/10"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      {/* Main Container Area */}
      <main className="max-w-7xl mx-auto px-8 py-12">
        {/* Title Block matching the exact typography weight from your screenshot */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 mb-10">
          <div>
            <h2 className="text-3xl font-extrabold tracking-tight text-white mb-2">
              Your Fusion Notebooks
            </h2>
            <p className="text-sm text-white">
              Choose an isolated notebook session or launch a fresh environment
              to start research.
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-[#007eff] to-[#00c6ff] text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 transition-all shadow-lg shadow-blue-500/20 active:scale-[0.98]"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            CREATE NEW NOTEBOOK
          </button>
        </div>

        {/* Sessions Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {notebooks.map((notebook) => (
            <div
              key={notebook.id}
              onClick={() => navigate(`/notebook/${notebook.id}`)}
              className="bg-[#111322] rounded-2xl border border-white/[0.04] hover:border-blue-500/30 transition-all duration-300 cursor-pointer group flex flex-col justify-between overflow-hidden relative shadow-xl shadow-black/20"
            >
              {/* Header Content Section of the Card */}
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-9 h-9 bg-white/5 text-blue-400 rounded-xl flex items-center justify-center group-hover:bg-blue-500/10 transition-colors">
                    <BookOpen className="w-4 h-4" />
                  </div>
                  <div className="text-[10px] font-semibold text-slate-400 bg-white/5 border border-white/5 px-2.5 py-1 rounded-lg flex items-center gap-1.5">
                    <Clock className="w-3 h-3 text-blue-400" />
                    {notebook.lastModified}
                  </div>
                </div>
                <h3 className="text-md font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
                  {notebook.name}
                </h3>
                <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed">
                  {notebook.description ||
                    "No unique parameters customized for this learning path yet."}
                </p>
              </div>

              {/* Lower Details / Features list styled just like your "Key Features" layout block */}
              <div className="px-6 pb-6 pt-4 bg-black/20 border-t border-white/[0.02] flex flex-col gap-4">
                

                <div className="border-t border-white/5 pt-3 flex items-center justify-between text-[10px] text-slate-500 font-medium">
                  <span>Created {notebook.createdAt}</span>
                  <span className="text-blue-400 font-bold opacity-0 group-hover:opacity-100 transition-opacity">
                    Open Hub →
                  </span>
                </div>
              </div>
            </div>
          ))}

          {/* Empty Layout Component Framework */}
          {notebooks.length === 0 && (
            <div className="col-span-full bg-[#111322] border border-dashed border-white/10 rounded-3xl flex flex-col items-center justify-center py-20 px-4">
              <div className="w-12 h-12 bg-white/5 text-slate-400 rounded-2xl flex items-center justify-center mb-4">
                <BookOpen className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-lg font-bold text-white mb-1">
                No active workspace nodes found
              </h3>
              <p className="text-xs text-slate-400 text-center max-w-sm mb-6 leading-relaxed">
                Your sandbox is clear. Initiate your first context pool to start
                using your conversational layout tools.
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-5 py-2.5 bg-gradient-to-r from-[#007eff] to-[#00c6ff] text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 transition-all"
              >
                Launch First Hub
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Modern Popover Dialog Frame Container */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-[#111322] rounded-2xl max-w-md w-full p-6 shadow-2xl border border-white/10 flex flex-col gap-5">
            <div>
              <h3 className="text-lg font-bold text-white tracking-tight">
                Create New Fusion Notebook
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Assign a clean context cluster parameters layout for this index.
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Notebook Identity Title
                </label>
                <input
                  type="text"
                  value={newNotebookName}
                  onChange={(e) => setNewNotebookName(e.target.value)}
                  placeholder="e.g., Quantum Mechanics Compendiums"
                  className="w-full text-sm px-4 py-3 bg-black/30 border border-white/5 rounded-xl text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500 transition-all"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Scope Focus Description{" "}
                  <span className="text-slate-500 font-normal">(Optional)</span>
                </label>
                <textarea
                  value={newNotebookDesc}
                  onChange={(e) => setNewNotebookDesc(e.target.value)}
                  placeholder="Summarize the core learning directives of this workspace..."
                  rows={3}
                  className="w-full text-sm px-4 py-3 bg-black/30 border border-white/5 rounded-xl text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500/50 focus:border-blue-500 resize-none transition-all"
                />
              </div>
            </div>

            <div className="flex gap-3 mt-1">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewNotebookName("");
                  setNewNotebookDesc("");
                }}
                className="flex-1 px-4 py-2.5 bg-white/5 hover:bg-white/10 text-slate-300 text-xs font-semibold rounded-xl border border-white/5 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateNotebook}
                disabled={!newNotebookName.trim()}
                className="flex-1 px-4 py-2.5 bg-gradient-to-r from-[#007eff] to-[#00c6ff] text-white text-xs font-bold rounded-xl transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Launch Hub
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState, useRef, useEffect } from "react";
import { SourcesSidebar } from "./SourcesSidebar";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";
import { useAuth, supabase } from "../contexts/AuthContext";
import { useNotebook } from "../contexts/NotebookContext";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, BookOpen, Trash2, RefreshCw, LogOut } from "lucide-react";

interface Source {
  id: string;
  name: string;
  type: string;
  pages?: number;
  url?: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export function FusionNotebookWorkspace() {
  const { user, logout } = useAuth();
  const { notebookId } = useParams();
  const navigate = useNavigate();

  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { setNotebook, currentNotebook } = useNotebook();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (!notebookId) return;

    const loadNotebook = async () => {
      const { data } = await supabase
        .from("notebooks")
        .select("notebook_id, name")
        .eq("notebook_id", notebookId)
        .single();

      if (data) setNotebook(data);
    };

    loadNotebook();
  }, [notebookId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchSources = async (): Promise<Source[]> => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const res = await fetch(
        "http://localhost:8000/api/notebooks/get_contents",
        {
          headers: {
            Authorization: `Bearer ${session?.access_token}`,
            "X-Notebook-Id": notebookId || "",
          },
        },
      );

      const data = await res.json();
      return data.sources || [];
    } catch {
      return [];
    }
  };

  useEffect(() => {
    if (!notebookId) return;
    fetchSources().then(setSources);
  }, [notebookId]);

  const fetchMessages = async (): Promise<Message[]> => {
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      const res = await fetch(
        "http://localhost:8000/api/notebooks/chat/messages",
        {
          headers: {
            Authorization: `Bearer ${session?.access_token}`,
            "X-Notebook-Id": notebookId || "",
          },
        },
      );

      const data = await res.json();

      return data.messages.map((msg: any, i: number) => ({
        id: i.toString(),
        role: msg.role,
        content: msg.content,
        timestamp: new Date(msg.timestamp).toLocaleTimeString(),
      }));
    } catch {
      return [];
    }
  };

  useEffect(() => {
    if (!notebookId) return;
    fetchMessages().then(setMessages);
  }, [notebookId]);

  const handleSendMessage = async (content: string) => {
    setIsTyping(true);

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      const res = await fetch(
        `http://localhost:8000/api/documents/query?q=${encodeURIComponent(
          content,
        )}`,
        {
          headers: {
            Authorization: `Bearer ${session?.access_token}`,
            "X-Notebook-Id": notebookId || "",
          },
        },
      );

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: data.results,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
      fetchMessages().then(setMessages);
    } catch (err) {
      console.error(err);
    } finally {
      setIsTyping(false);
    }
  };

  const handleResetChat = async () => {
    const confirmReset = window.confirm(
      "Are you sure you want to reset the chat?\n\nAll conversation history will be lost.",
    );

    if (!confirmReset) return;

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      await fetch("http://localhost:8000/api/notebooks/reset_chat", {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
          "X-User-Id": session?.user?.id || "",
          "X-Notebook-Id": notebookId || "",
        },
      });

      setMessages([]);
    } catch (err) {
      console.error(err);
      alert("Failed to reset chat");
    }
  };

  const handleResetSources = async () => {
    const confirmReset = window.confirm(
      "Are you sure you want to remove ALL sources?\n\nThis will erase all memory and documents.",
    );

    if (!confirmReset) return;

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      await fetch("http://localhost:8000/api/notebooks/delete_contents", {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
          "X-User-Id": session?.user?.id || "",
          "X-Notebook-Id": notebookId || "",
        },
      });

      setSources([]);
    } catch (err) {
      console.error(err);
      alert("Failed to reset sources");
    }
  };

  const handleRemoveSource = async (id: string) => {
    const source = sources.find((s) => s.id === id);
    if (!source) return;

    const confirmDelete = window.confirm(
      `Are you sure you want to remove "${source.name}"?\n\nThis will not be used in future conversations.`,
    );

    if (!confirmDelete) return;
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();

      await fetch(
        `http://localhost:8000/api/notebooks/delete_source?source_name=${encodeURIComponent(
          source.name,
        )}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${session?.access_token}`,
            "X-User-Id": session?.user?.id || "",
            "X-Notebook-Id": notebookId || "",
          },
        },
      );

      fetchSources().then(setSources);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex h-screen bg-[#0d0e1b] text-[#94a3b8] font-sans antialiased selection:bg-blue-500/30 selection:text-white overflow-hidden">
      {/* SIDEBAR ANCHOR WRAPPER */}
      <div className="border-r border-white/5 bg-[#111322] flex flex-col shrink-0">
        <SourcesSidebar
          sources={sources}
          onAddSource={(source) =>
            setSources((prev) => [
              ...prev,
              { id: Date.now().toString(), ...source },
            ])
          }
          onRemoveSource={handleRemoveSource}
        />
      </div>

      {/* CORE WORKSPACE AREA */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0d0e1b]">
        {/* TOP HEADER BAR */}
        <header className="bg-[#111322]/60 border-b border-white/5 sticky top-0 z-40 backdrop-blur-md shrink-0">
          <div className="px-8 py-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-4 min-w-0">
              <button
                onClick={() => navigate("/notebooks")}
                className="p-2 hover:bg-white/5 rounded-xl text-slate-400 hover:text-white transition-all shrink-0"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-9 h-9 bg-gradient-to-tr from-blue-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20 shrink-0">
                  <BookOpen className="w-4 h-4 text-white" />
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <h1 className="text-md font-bold tracking-tight text-white leading-none truncate">
                      {currentNotebook?.name || "Fusion Notebook Workspace"}
                    </h1>
                    <span className="text-[10px] bg-blue-500/10 text-blue-400 px-1.5 py-0.5 rounded font-medium border border-blue-500/20 shrink-0">
                      Active Session
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 truncate">
                    FusionAI Multi-source Research Environment
                  </p>
                </div>
              </div>
            </div>

            {/* ACTION TOOLKIT CONTROL GRID */}
            <div className="flex items-center gap-3 shrink-0">
              <button
                onClick={handleResetChat}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-amber-400 bg-amber-500/5 hover:bg-amber-500/10 border border-amber-500/20 rounded-xl transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Reset Chat</span>
              </button>

              <button
                onClick={handleResetSources}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-rose-400 bg-rose-500/5 hover:bg-rose-500/10 border border-rose-500/20 rounded-xl transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Clear Contents</span>
              </button>

              <div className="h-4 w-px bg-white/10 mx-1 hidden md:block" />

              {/* USER BRAND CORNER OVERLAY */}
              <div className="hidden md:flex items-center gap-3 px-3 py-1.5 rounded-xl bg-[#111322] border border-white/5">
                <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-emerald-500 to-teal-600 flex items-center justify-center text-white text-[10px] font-bold overflow-hidden ring-1 ring-white/10">
                  {user?.avatar ? (
                    <img src={user.avatar} alt={user.name} className="w-full h-full object-cover" />
                  ) : (
                    <span>{user?.name?.charAt(0) || "S"}</span>
                  )}
                </div>
                <div className="text-right">
                  <p className="text-[11px] font-semibold text-white leading-tight">
                    {user?.name || "Shourya Mishra"}
                  </p>
                </div>
              </div>

              <button
                onClick={logout}
                className="p-2 md:px-4 md:py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-[#111322] hover:bg-[#16192e] rounded-xl transition-all border border-white/10"
                title="Sign out"
              >
                <LogOut className="w-4 h-4 md:hidden" />
                <span className="hidden md:inline">Sign out</span>
              </button>
            </div>
          </div>
        </header>

        {/* MESSAGES CORE FRAMEWORK */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6 scrollbar-thin scrollbar-thumb-white/5">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <EmptyState />
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((m) => (
                <ChatMessage key={m.id} {...m} />
              ))}
            </div>
          )}
          {isTyping && (
            <div className="max-w-4xl mx-auto pl-4 flex items-center gap-2 text-xs font-mono text-cyan-400/80 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 block animate-bounce" />
              Synthesizing response node...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* INPUT SHELF CONTAINER */}
        <div className="p-8 bg-gradient-to-t from-[#0d0e1b] via-[#0d0e1b] to-transparent shrink-0">
          <div className="max-w-4xl mx-auto">
            <ChatInput onSend={handleSendMessage} disabled={isTyping} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default FusionNotebookWorkspace;
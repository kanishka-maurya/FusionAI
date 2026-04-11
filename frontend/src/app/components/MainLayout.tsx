import { useState, useRef, useEffect } from "react";
import { SourcesSidebar } from "./SourcesSidebar";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";
import { useAuth, supabase } from "../contexts/AuthContext";
import { useNotebook } from "../contexts/NotebookContext";
import { useParams } from "react-router-dom";

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

function MainLayout() {
  const { user, logout } = useAuth();
  const { notebookId } = useParams();

  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { setNotebook } = useNotebook();

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
      console.log(notebookId)
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
    "Are you sure you want to reset the chat?\n\nAll conversation history will be lost."
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
    "Are you sure you want to remove ALL sources?\n\nThis will erase all memory and documents."
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
    console.log("Deleting with name:", source.name);
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

      // refresh sources
      fetchSources().then(setSources);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
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

      <div className="flex-1 flex flex-col">
        {/* HEADER */}
        <header className="bg-white border-b px-6 py-4 flex justify-between items-center">
          <h1 className="text-xl font-semibold">NotebookLM Chat</h1>

          <div className="flex gap-3">
            {/* RESET CHAT */}
            <button
              onClick={handleResetChat}
              className="bg-yellow-500 text-white px-3 py-1 rounded hover:bg-yellow-600"
            >
              Reset Chat
            </button>

            {/* RESET CONTENT */}
            <button
              onClick={handleResetSources}
              className="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600"
            >
              Reset Contents
            </button>

            <button onClick={logout}>Sign Out</button>
          </div>
        </header>

        {/* CHAT */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            messages.map((m) => <ChatMessage key={m.id} {...m} />)
          )}
          {isTyping && <div className="p-4">Typing...</div>}
          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isTyping} />
      </div>
    </div>
  );
}

export default MainLayout;

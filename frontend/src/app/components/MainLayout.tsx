import { useState, useRef, useEffect } from "react";
import { SourcesSidebar } from "./SourcesSidebar";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";
import { useAuth, supabase } from "../contexts/AuthContext"; 

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
  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  const handleAddSource = (source: { name: string; type: string; url?: string; pages?: number }) => {
    const newSource: Source = {
      id: Date.now().toString(),
      ...source,
    };
    setSources([...sources, newSource]);
  };

  const handleRemoveSource = (id: string) => {
    setSources(sources.filter((source) => source.id !== id));
  };

  const handleSendMessage = async (content: string) => {
    setIsTyping(true); 
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const { data: { session } } = await supabase.auth.getSession();
      

      const res = await fetch(
        `http://localhost:8000/api/documents/query?q=${encodeURIComponent(content)}`,
        {
          headers: {
            'Authorization': `Bearer ${session?.access_token}`,
          }
        }
      );

      if (!res.ok) throw new Error("Query failed");

      const data = await res.json();
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.results, 
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
    } finally {
      setIsTyping(false); 
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <SourcesSidebar
        sources={sources}
        onAddSource={handleAddSource}
        onRemoveSource={handleRemoveSource}
      />

      <div className="flex-1 flex flex-col">
        {/* Updated Header with Username */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-semibold">NotebookLM Chat</h1>
            <p className="text-sm text-gray-600">
              Welcome back, <span className="font-medium text-blue-600">{user?.name || 'User'}</span>
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-medium text-gray-900">{user?.name}</p>
              <p className="text-xs text-gray-500">{user?.email}</p>
            </div>
            {user?.avatar && (
              <img src={user.avatar} alt="Profile" className="w-10 h-10 rounded-full border border-gray-200" />
            )}
            <button 
              onClick={logout}
              className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-md transition"
            >
              Sign Out
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? <EmptyState /> : (
            <div className="max-w-4xl mx-auto w-full">
              {messages.map((message) => (
                <ChatMessage key={message.id} {...message} />
              ))}
              {isTyping && <div className="p-6">Typing...</div>}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isTyping} />
      </div>
    </div>
  );
}

export default MainLayout;

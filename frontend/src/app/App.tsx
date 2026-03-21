import { useState, useRef, useEffect } from "react";
import { SourcesSidebar } from "./components/SourcesSidebar";
import { ChatMessage } from "./components/ChatMessage";
import { ChatInput } from "./components/ChatInput";
import { EmptyState } from "./components/EmptyState";

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

function App() {
  const [sources, setSources] = useState<Source[]>([
    {
      id: "1",
      name: "Introduction to Machine Learning.pdf",
      type: "PDF Document",
      pages: 45,
    },
    {
      id: "2",
      name: "Research Paper - Neural Networks.pdf",
      type: "PDF Document",
      pages: 28,
    },
  ]);

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

  const generateResponse = (userMessage: string): string => {
    const responses = [
      `Based on your sources, I can help explain that topic. The documents you've uploaded contain relevant information about ${userMessage.toLowerCase()}.`,
      `That's an interesting question! From analyzing your sources, here are the key points: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.`,
      `According to the research papers you've provided, neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.`,
      `Let me summarize the main concepts from your documents: The fundamental principle involves training models on data, allowing them to recognize patterns and make predictions on new, unseen data.`,
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleSendMessage = (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: generateResponse(content),
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <SourcesSidebar
        sources={sources}
        onAddSource={handleAddSource}
        onRemoveSource={handleRemoveSource}
      />

      <div className="flex-1 flex flex-col">
        <header className="bg-white border-b border-gray-200 px-6 py-4">
          <h1 className="text-xl font-semibold">NotebookLM Chat</h1>
          <p className="text-sm text-gray-600">
            AI-powered research assistant
          </p>
        </header>

        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="max-w-4xl mx-auto w-full">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  timestamp={message.timestamp}
                />
              ))}
              {isTyping && (
                <div className="flex gap-4 p-6 bg-gray-50">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-purple-600">
                    <div className="w-5 h-5 text-white">...</div>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <ChatInput onSend={handleSendMessage} disabled={isTyping} />
      </div>
    </div>
  );
}

export default App;
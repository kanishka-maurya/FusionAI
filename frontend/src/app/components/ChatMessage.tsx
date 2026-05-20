import { Bot, User, Cpu } from "lucide-react";
import { useAuth, supabase } from "../contexts/AuthContext";
interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  const isUser = role === "user";
  const { user, logout } = useAuth();
  return (
    <div 
      className={`flex gap-4 p-5 rounded-2xl border transition-all duration-200 ${
        isUser 
          ? "bg-white/[0.02] border-white/[0.04] shadow-md shadow-black/10" 
          : "bg-gradient-to-r from-blue-500/[0.03] to-cyan-500/[0.01] border-blue-500/10 shadow-lg shadow-blue-950/5"
      }`}
    >
      {/* AVATAR NODE DESIGN */}
      <div className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center border ${
        isUser 
          ? "bg-slate-800 border-slate-700 shadow-sm" 
          : "bg-gradient-to-tr from-blue-600 to-cyan-500 border-blue-400/20 shadow-md shadow-blue-500/10"
      }`}>
        {isUser ? (
          <User className="w-4 h-4 text-slate-300" />
        ) : (
          <Cpu className="w-4 h-4 text-white animate-pulse" />
        )}
      </div>
      
      {/* CONTENT BODY ELEMENT */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <div className="flex items-center gap-2">
            <span className={`text-xs font-bold tracking-wide uppercase ${
              isUser ? "text-slate-300" : "text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400"
            }`}>
              {isUser ? user?.name?.split(" ")[0] : "FusionAI Core"}
            </span>
            {!isUser && (
              <span className="text-[9px] bg-cyan-500/10 text-cyan-400 px-1.5 py-0.2 rounded font-mono border border-cyan-500/20">
                Synthesis Node
              </span>
            )}
          </div>
          {timestamp && (
            <span className="text-[10px] font-mono text-slate-500 tracking-tight">{timestamp}</span>
          )}
        </div>
        
        {/* TEXT VALUE DISPATCHER */}
        <div className={`text-sm whitespace-pre-wrap leading-relaxed prose prose-invert max-w-none ${
          isUser ? "text-slate-300 font-medium" : "text-slate-100 font-normal"
        }`}>
          {content}
        </div>
      </div>
    </div>
  );
}
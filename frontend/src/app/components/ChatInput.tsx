import { Send, Terminal } from "lucide-react";
import { useState, KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="bg-[#111322]/40 backdrop-blur-md border border-white/5 p-4 rounded-2xl shadow-xl shadow-black/20">
      <div className="max-w-4xl mx-auto">
        <div className="relative flex items-center gap-2">
          {/* VISUAL TERMINAL INDICATOR */}
          <div className="absolute left-4 text-slate-500 pointer-events-none flex items-center justify-center">
            <Terminal className="w-4 h-4 text-slate-500" />
          </div>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Cross-examine loaded knowledge pools..."
            disabled={disabled}
            className="w-full resize-none rounded-xl border border-white/5 bg-black/30 pl-11 pr-14 py-3.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/40 focus:ring-1 focus:ring-blue-500/20 disabled:bg-white/[0.01] disabled:text-slate-600 disabled:cursor-not-allowed min-h-[48px] max-h-32 transition-all scrollbar-none"
            rows={1}
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || disabled}
            className="absolute right-2 p-2 bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-lg hover:opacity-90 active:scale-95 disabled:from-white/5 disabled:to-white/5 disabled:text-slate-600 disabled:cursor-not-allowed disabled:scale-100 transition-all border border-white/5"
            title="Execute model evaluation query"
          >
            <Send className="w-3.5 h-3.5 stroke-[2.5]" />
          </button>
        </div>
        
        <div className="flex items-center justify-center gap-4 mt-2 text-[10px] font-mono tracking-tight text-slate-500">
          <span>[Enter] Submit Query Node</span>
          <span className="text-slate-700">•</span>
          <span>[Shift + Enter] Breakline Code</span>
        </div>
      </div>
    </div>
  );
}
import { MessageSquare, Lightbulb, FileSearch, Sparkles, Terminal } from "lucide-react";

export function EmptyState() {
  const suggestions = [
    {
      icon: FileSearch,
      text: "Synthesize global summaries and key milestones from loaded indices.",
      color: "text-blue-400 bg-blue-500/10 border-blue-500/20",
    },
    {
      icon: Lightbulb,
      text: "Extract hidden insights and cross-functional trends across sources.",
      color: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    },
    {
      icon: Sparkles,
      text: "Compare contradictory data layers and isolate variable core vectors.",
      color: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20",
    },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8 bg-[#0d0e1b]">
      <div className="max-w-xl text-center space-y-10 animate-fadeIn">
        {/* HERO ICON AND HEADER CLUSTER */}
        <div className="space-y-4">
          <div className="relative w-16 h-16 mx-auto flex items-center justify-center rounded-2xl bg-gradient-to-tr from-blue-600 to-cyan-500 shadow-xl shadow-blue-500/10 border border-blue-400/20 group">
            <MessageSquare className="w-7 h-7 text-white" />
            <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-400"></span>
            </span>
          </div>
          
          <div className="space-y-2">
            <h1 className="text-xl font-bold tracking-tight text-white sm:text-2xl">
              Query the <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400">Fusion Workspace Hub</span>
            </h1>
            <p className="text-xs text-slate-400 leading-relaxed max-w-md mx-auto">
              Mount your local files, video references, or transcript assets onto the sidebar to build a responsive context engine.
            </p>
          </div>
        </div>

        {/* INTERACTIVE SUGGESTION FRAMEWORK */}
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-2 text-[10px] font-mono tracking-wider text-slate-500 uppercase">
            <Terminal className="w-3.5 h-3.5 text-blue-500" />
            <span>Recommended Evaluation Triggers</span>
          </div>
          
          <div className="grid gap-3 text-left">
            {suggestions.map((suggestion, index) => {
              const Icon = suggestion.icon;
              return (
                <div
                  key={index}
                  className="group flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-blue-500/20 hover:bg-white/[0.03] active:scale-[0.99] transition-all duration-200 cursor-pointer"
                >
                  <div className={`w-9 h-9 rounded-xl flex items-center justify-center border shrink-0 group-hover:scale-105 transition-transform ${suggestion.color}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-medium text-slate-300 group-hover:text-slate-100 transition-colors leading-normal">
                    {suggestion.text}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
import { FileText, Plus, X, Youtube, Mic, ClipboardType, Layers } from "lucide-react";
import { useState } from "react";
import { AddSourceModal } from "./AddSourceModal";

interface Source {
  id: string;
  name: string;
  type: string;
  pages?: number;
  url?: string;
}

interface SourcesSidebarProps {
  sources: Source[];
  onAddSource: (source: {
    name: string;
    type: string;
    url?: string;
    pages?: number;
  }) => void;
  onRemoveSource: (id: string) => void;
}

export function SourcesSidebar({
  sources = [],
  onAddSource,
  onRemoveSource,
}: SourcesSidebarProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const getSourceIcon = (type: string) => {
    if (type.includes("YouTube"))
      return (
        <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center shrink-0 border border-red-500/20">
          <Youtube className="w-4 h-4 text-red-400" />
        </div>
      );
    if (type.includes("Audio"))
      return (
        <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center shrink-0 border border-purple-500/20">
          <Mic className="w-4 h-4 text-purple-400" />
        </div>
      );
    if (type.includes("Text"))
      return (
        <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center shrink-0 border border-emerald-500/20">
          <ClipboardType className="w-4 h-4 text-emerald-400" />
        </div>
      );
    return (
      <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center shrink-0 border border-blue-500/20">
        <FileText className="w-4 h-4 text-blue-400" />
      </div>
    );
  };

  return (
    <>
      <div className="w-80 bg-[#111322] flex flex-col h-full border-r border-white/5">
        {/* SIDEBAR HEADER CONTAINER */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h2 className="font-bold text-sm uppercase tracking-wider text-slate-200">
              Context Sources
            </h2>
            <span className="ml-auto text-[10px] font-mono bg-white/5 text-slate-400 px-2 py-0.5 rounded-md border border-white/5">
              {sources.length} Total
            </span>
          </div>

          <button
            onClick={() => setIsModalOpen(true)}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold tracking-wide rounded-xl hover:opacity-90 active:scale-[0.98] transition-all shadow-md shadow-blue-500/10"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            INSERT SOURCE DATA
          </button>
        </div>

        {/* INDEXED DATA SOURCE LIST */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5 scrollbar-thin scrollbar-thumb-white/5">
          {sources.length === 0 ? (
            <div className="text-center text-slate-500 py-12 px-4 flex flex-col items-center justify-center h-full">
              <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-center mb-4">
                <FileText className="w-5 h-5 text-slate-400" />
              </div>
              <h3 className="text-xs font-bold text-slate-300 mb-1">No operational data loaded</h3>
              <p className="text-[11px] text-slate-400 leading-relaxed max-w-[200px] mx-auto">
                Seed this workspace hub with files, audio links, or clips to enable dynamic indexing answers.
              </p>
            </div>
          ) : (
            sources.map((source) => (
              <div
                key={source.id}
                className="group relative p-3 bg-white/[0.02] border border-white/[0.03] rounded-xl hover:bg-white/[0.05] hover:border-blue-500/20 transition-all duration-200"
              >
                <div className="flex items-center gap-3">
                  {getSourceIcon(source.type)}

                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-xs text-slate-200 truncate group-hover:text-blue-400 transition-colors">
                      {source.name}
                    </h3>
                    <p className="text-[10px] font-medium text-slate-400 mt-0.5 flex items-center gap-1">
                      <span className="uppercase tracking-tight text-[9px] text-slate-500">
                        {source.type.split(" ")[0]}
                      </span>
                      {source.pages && (
                        <>
                          <span className="text-slate-600">•</span>
                          <span>{source.pages} structural pages</span>
                        </>
                      )}
                    </p>
                  </div>

                  {/* ACTION CORNER: REMOVE ANCHOR NODE */}
                  <button
                    onClick={() => onRemoveSource(source.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity duration-150 p-1.5 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 rounded-lg"
                    title="Remove source parameter"
                  >
                    <X className="w-3.5 h-3.5 text-rose-400" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <AddSourceModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAddSource={onAddSource}
      />
    </>
  );
}
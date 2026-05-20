import { useState } from "react";
import {
  X,
  FileText,
  Youtube,
  Mic,
  ClipboardType,
  WebhookIcon,
  ArrowLeft,
  Sparkles,
} from "lucide-react";
import { supabase } from "../contexts/AuthContext";
import { useNotebook } from "../contexts/NotebookContext";

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddSource: (source: {
    name: string;
    type: string;
    url?: string;
    pages?: number;
  }) => void;
}

type SourceType = "document" | "youtube" | "audio" | "text" | "web" | null;

export function AddSourceModal({
  isOpen,
  onClose,
  onAddSource,
}: AddSourceModalProps) {
  const [selectedType, setSelectedType] = useState<SourceType>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [webUrl, setWebUrl] = useState("");
  const [copiedText, setCopiedText] = useState("");
  const [fileName, setFileName] = useState("");
  
  // Destructured 'notebook' to match your application's context schema perfectly
  const { currentNotebook } = useNotebook();

  if (!isOpen) return null;

  const handleAudioUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    type: "audio",
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/api/audio/upload", {
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Notebook-Id": currentNotebook?.notebook_id || "",
        },
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      onAddSource({
        name: file.name,
        type: type === "audio" ? "Audio File" : "Unknown",
      });

      handleClose();
    } catch (err) {
      console.error(err);
      alert("Audio upload failed");
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    type: "document" | "audio",
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      if (!session) {
        alert("Please log in first!");
        return;
      }
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/api/documents/upload", {
        headers: {
          Authorization: `Bearer ${token}`,
          "X-Notebook-Id": currentNotebook?.notebook_id || "",
        },
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      onAddSource({
        name: file.name,
        type: type === "document" ? "PDF Document" : "Audio File",
      });

      handleClose();
    } catch (err) {
      console.error(err);
      alert("File upload failed");
    }
  };

  const handleYoutubeSubmit = async () => {
    if (!youtubeUrl.trim()) return;

    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch(
        `http://localhost:8000/api/youtube/process_video_link?video_link=${encodeURIComponent(
          youtubeUrl,
        )}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "X-Notebook-Id": currentNotebook?.notebook_id || "",
          },
          method: "POST",
        },
      );

      if (!res.ok) throw new Error("Failed to process Youtube Link");

      onAddSource({
        name: youtubeUrl,
        type: "YouTube",
      });

      handleClose();
    } catch (err) {
      console.error(err);
      alert("YouTube processing failed");
    }
  };

  const handleWebSubmit = async () => {
    if (!webUrl.trim()) return;
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch(
        `http://localhost:8000/api/web/web_upload?url=${encodeURIComponent(
          webUrl,
        )}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "X-Notebook-Id": currentNotebook?.notebook_id || "",
          },
          method: "POST",
        },
      );

      if (!res.ok) throw new Error("Failed to process Web URL");

      onAddSource({
        name: webUrl,
        type: "Web URL",
      });

      handleClose();
    } catch (err) {
      console.error(err);
      alert("Web URL processing failed");
    }
  };

  const handleTextSubmit = async () => {
    if (!copiedText.trim()) return;

    const preview = copiedText.substring(0, 30);
    try {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;
      const res = await fetch("http://localhost:8000/api/text/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
          "X-Notebook-Id": currentNotebook?.notebook_id || "",
        },
        body: JSON.stringify({
          fileName: fileName,
          copiedText: copiedText,
        }),
      });
      if (!res.ok) {
        alert("Text processing failed");
        return;
      }
    } catch (err) {
      console.error(err);
      alert("Text processing failed");
      return;
    }

    onAddSource({
      name: fileName || `Text: ${preview}...`,
      type: "Copied Text",
    });

    handleClose();
  };

  const handleClose = () => {
    setSelectedType(null);
    setYoutubeUrl("");
    setWebUrl("");
    setCopiedText("");
    setFileName("");
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black/85 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fadeIn"
      onClick={handleClose}
    >
      <div
        className="bg-[#111322] rounded-2xl max-w-md w-full border border-white/10 shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* MODAL WINDOW HEADER */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/5">
          <div>
            <h2 className="text-md font-bold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              Feed Context Knowledge
            </h2>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Select an ingestion pipeline for this workspace cluster.
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-1.5 hover:bg-white/5 text-slate-400 hover:text-white rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* CONTAINER WORKSPACE BODY */}
        <div className="p-6">
          {!selectedType ? (
            <div className="space-y-3">
              {/* DOCUMENT PIPELINE BUTTON */}
              <button
                onClick={() => document.getElementById("document-upload")?.click()}
                className="w-full flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-blue-500/40 hover:bg-blue-500/[0.02] text-left transition-all duration-200 group"
              >
                <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center border border-blue-500/20 group-hover:bg-blue-500/20 transition-colors">
                  <FileText className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 group-hover:text-blue-400 transition-colors">
                    Upload Document File
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">PDF, DOCX, or operational TXT layout nodes</p>
                </div>
              </button>

              <input
                id="document-upload"
                type="file"
                accept=".pdf,.doc,.docx,.txt"
                onChange={(e) => handleFileUpload(e, "document")}
                className="hidden"
              />

              {/* YOUTUBE LINK INTERFACE */}
              <button
                onClick={() => setSelectedType("youtube")}
                className="w-full flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-red-500/40 hover:bg-red-500/[0.02] text-left transition-all duration-200 group"
              >
                <div className="w-10 h-10 rounded-xl bg-red-500/10 flex items-center justify-center border border-red-500/20 group-hover:bg-red-500/20 transition-colors">
                  <Youtube className="w-5 h-5 text-red-400" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 group-hover:text-red-400 transition-colors">
                    YouTube Stream Capture
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Parse dynamic summaries out of video transcripts</p>
                </div>
              </button>

              {/* AUDIO RECORDING / TRACK PIPELINE */}
              {/* <button
                onClick={() => document.getElementById("audio-upload")?.click()}
                className="w-full flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-purple-500/40 hover:bg-purple-500/[0.02] text-left transition-all duration-200 group"
              >
                <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center border border-purple-500/20 group-hover:bg-purple-500/20 transition-colors">
                  <Mic className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 group-hover:text-purple-400 transition-colors">
                    Audio Structural Payload
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Process voice logs or conversational tracks</p>
                </div>
              </button> */}

              <input
                id="audio-upload"
                type="file"
                accept=".mp3,.wav,.m4a"
                onChange={(e) => handleAudioUpload(e, "audio")}
                className="hidden"
              />

              {/* RAW COPIED CLIPS TEXT */}
              <button
                onClick={() => setSelectedType("text")}
                className="w-full flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-emerald-500/40 hover:bg-emerald-500/[0.02] text-left transition-all duration-200 group"
              >
                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20 group-hover:bg-emerald-500/20 transition-colors">
                  <ClipboardType className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 group-hover:text-emerald-400 transition-colors">
                    Direct Text Custom Injection
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Paste documentation segments manually</p>
                </div>
              </button>

              {/* EXTERNAL WEB INDEX ARCHITECTURE */}
              <button
                onClick={() => setSelectedType("web")}
                className="w-full flex items-center gap-4 p-4 bg-white/[0.01] border border-white/5 rounded-xl hover:border-amber-500/40 hover:bg-amber-500/[0.02] text-left transition-all duration-200 group"
              >
                <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20 group-hover:bg-amber-500/20 transition-colors">
                  <WebhookIcon className="w-5 h-5 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-xs font-bold text-slate-200 group-hover:text-amber-400 transition-colors">
                    Web URL Context Node
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Scrape content pools directly off external links</p>
                </div>
              </button>
            </div>
          ) : selectedType === "youtube" ? (
            <InputUI
              label="YouTube Clip Location String"
              placeholder="e.g., https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              setValue={setYoutubeUrl}
              onSubmit={handleYoutubeSubmit}
              onBack={() => setSelectedType(null)}
              themeColor="border-red-500/50 focus:border-red-500 focus:ring-red-500/20"
              btnColor="bg-gradient-to-r from-red-600 to-rose-500"
            />
          ) : selectedType === "web" ? (
            <InputUI
              label="Target Web URL Resource"
              placeholder="e.g., https://wikipedia.org/wiki/Quantum_mechanics"
              value={webUrl}
              setValue={setWebUrl}
              onSubmit={handleWebSubmit}
              onBack={() => setSelectedType(null)}
              themeColor="border-amber-500/50 focus:border-amber-500 focus:ring-amber-500/20"
              btnColor="bg-gradient-to-r from-amber-600 to-yellow-500"
            />
          ) : selectedType === "text" ? (
            <div className="space-y-4">
              <button
                onClick={() => setSelectedType(null)}
                className="inline-flex items-center gap-1 text-[11px] font-bold text-slate-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" /> Return To Directory
              </button>

              <div className="space-y-3">
                <div>
                  <label className="block text-[11px] font-bold text-slate-300 uppercase tracking-wide mb-1.5">
                    Data Cluster Title Identity
                  </label>
                  <input
                    placeholder="e.g., Condensed Lecture Notes Node"
                    value={fileName}
                    onChange={(e) => setFileName(e.target.value)}
                    className="w-full text-xs px-4 py-3 bg-black/30 border border-white/5 rounded-xl text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-all"
                  />
                </div>

                <div>
                  <label className="block text-[11px] font-bold text-slate-300 uppercase tracking-wide mb-1.5">
                    Text Raw Value Stream
                  </label>
                  <textarea
                    placeholder="Provide the source raw materials right here..."
                    value={copiedText}
                    onChange={(e) => setCopiedText(e.target.value)}
                    rows={6}
                    className="w-full text-xs px-4 py-3 bg-black/30 border border-white/5 rounded-xl text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 resize-none transition-all"
                  />
                </div>
              </div>

              <button
                onClick={handleTextSubmit}
                disabled={!copiedText.trim() || !fileName.trim()}
                className="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-teal-500 text-white text-xs font-bold rounded-xl tracking-wide hover:opacity-90 transition-all disabled:opacity-20 disabled:cursor-not-allowed shadow-md shadow-emerald-500/10"
              >
                DEPLOY CONTEXT BLOCK
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

interface InputUIProps {
  label: string;
  placeholder: string;
  value: string;
  setValue: (val: string) => void;
  onSubmit: () => void;
  onBack: () => void;
  themeColor: string;
  btnColor: string;
}

function InputUI({
  label,
  placeholder,
  value,
  setValue,
  onSubmit,
  onBack,
  themeColor,
  btnColor,
}: InputUIProps) {
  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="inline-flex items-center gap-1 text-[11px] font-bold text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Return To Directory
      </button>

      <div>
        <label className="block text-[11px] font-bold text-slate-300 uppercase tracking-wide mb-1.5">
          {label}
        </label>
        <input
          type="url"
          placeholder={placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className={`w-full text-xs px-4 py-3 bg-black/30 border border-white/5 rounded-xl text-white placeholder-slate-600 focus:outline-none focus:ring-1 transition-all ${themeColor}`}
        />
      </div>

      <button
        onClick={onSubmit}
        disabled={!value.trim()}
        className={`w-full py-2.5 text-white text-xs font-bold rounded-xl tracking-wide hover:opacity-90 transition-all disabled:opacity-20 disabled:cursor-not-allowed shadow-lg ${btnColor}`}
      >
        MOUNT PARAMETER NODE
      </button>
    </div>
  );
}
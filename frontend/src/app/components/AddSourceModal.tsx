import { useState } from "react";
import {
  X,
  FileText,
  Youtube,
  Mic,
  ClipboardType,
  WebhookIcon,
} from "lucide-react";

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

  if (!isOpen) return null;

  const handleAudioUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    type: "audio",
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const formData = new FormData();
      formData.append("file", file);
      console.log(file.name);
      const res = await fetch("http://localhost:8000/api/audio/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      onAddSource({
        name: file.name,
        type: type === "audio" ? "Audio File" : "Unknown",
      });

      console.log("Processed:", data);
      handleClose();
    } catch (err) {
      console.log(err);
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
      const formData = new FormData();
      formData.append("file", file);
      console.log(formData.get("file"));
      const res = await fetch("http://localhost:8000/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      onAddSource({
        name: file.name,
        type: type === "document" ? "PDF Document" : "Audio File",
      });

      console.log("Processed:", data);
      handleClose();
    } catch (err) {
      console.error(err);
      alert("File upload failed");
    }
  };

  const handleYoutubeSubmit = async () => {
    if (!youtubeUrl.trim()) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/youtube/process_video_link?video_link=${encodeURIComponent(
          youtubeUrl,
        )}`,
        { method: "POST" },
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
    console.log(webUrl);
    try {
      const res = await fetch(
        `http://localhost:8000/api/web/web_upload?url=${encodeURIComponent(
          webUrl,
        )}`,
        {
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
      const res = await fetch("http://localhost:8000/api/text/process", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
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
      console.log(err);
      alert("Text processing failed");
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
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={handleClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* HEADER */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold">Add Source</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        {/* BODY */}
        <div className="p-6">
          {!selectedType ? (
            <div className="space-y-3">
              {/* DOCUMENT */}
              <button
                onClick={() =>
                  document.getElementById("document-upload")?.click()
                }
                className="w-full flex items-center gap-4 p-4 border rounded-lg hover:border-blue-500 hover:bg-blue-50"
              >
                <FileText className="w-6 h-6 text-blue-600" />
                <div>
                  <h3>Upload Document</h3>
                </div>
              </button>

              <input
                id="document-upload"
                type="file"
                accept=".pdf,.doc,.docx,.txt"
                onChange={(e) => handleFileUpload(e, "document")}
                className="hidden"
              />

              {/* YOUTUBE */}
              <button
                onClick={() => setSelectedType("youtube")}
                className="w-full flex items-center gap-4 p-4 border rounded-lg hover:border-red-500 hover:bg-red-50"
              >
                <Youtube className="w-6 h-6 text-red-600" />
                <div>
                  <h3>YouTube Link</h3>
                </div>
              </button>

              {/* AUDIO */}
              <button
                onClick={() => document.getElementById("audio-upload")?.click()}
                className="w-full flex items-center gap-4 p-4 border rounded-lg hover:border-purple-500 hover:bg-purple-50"
              >
                <Mic className="w-6 h-6 text-purple-600" />
                <div>
                  <h3>Audio File</h3>
                </div>
              </button>

              <input
                id="audio-upload"
                type="file"
                accept=".mp3,.wav,.m4a"
                onChange={(e) => handleAudioUpload(e, "audio")}
                className="hidden"
              />

              {/* TEXT */}
              <button
                onClick={() => setSelectedType("text")}
                className="w-full flex items-center gap-4 p-4 border rounded-lg hover:border-green-500 hover:bg-green-50"
              >
                <ClipboardType className="w-6 h-6 text-green-600" />
                <div>
                  <h3>Copied Text</h3>
                </div>
              </button>

              {/* WEB URL */}
              <button
                onClick={() => setSelectedType("web")}
                className="w-full flex items-center gap-4 p-4 border rounded-lg hover:border-yellow-500 hover:bg-yellow-50"
              >
                <WebhookIcon className="w-6 h-6 text-yellow-600" />
                <div>
                  <h3>Web URL</h3>
                </div>
              </button>
            </div>
          ) : selectedType === "youtube" ? (
            <InputUI
              label="YouTube URL"
              value={youtubeUrl}
              setValue={setYoutubeUrl}
              onSubmit={handleYoutubeSubmit}
              onBack={() => setSelectedType(null)}
            />
          ) : selectedType === "web" ? (
            <InputUI
              label="Website URL"
              value={webUrl}
              setValue={setWebUrl}
              onSubmit={handleWebSubmit}
              onBack={() => setSelectedType(null)}
            />
          ) : selectedType === "text" ? (
            <div className="space-y-4">
              <button onClick={() => setSelectedType(null)}>← Back</button>

              <input
                placeholder="Name"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                className="w-full border p-2 rounded"
              />

              <textarea
                value={copiedText}
                onChange={(e) => setCopiedText(e.target.value)}
                rows={5}
                className="w-full border p-2 rounded"
              />

              <button
                onClick={handleTextSubmit}
                className="bg-blue-600 text-white px-4 py-2 rounded"
              >
                Add
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// 🔥 Reusable input component
function InputUI({ label, value, setValue, onSubmit, onBack }: any) {
  return (
    <div className="space-y-4">
      <button onClick={onBack}>← Back</button>

      <div>
        <label className="block text-sm mb-2">{label}</label>
        <input
          type="url"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="w-full border p-2 rounded"
        />
      </div>

      <button
        onClick={onSubmit}
        disabled={!value.trim()}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Add
      </button>
    </div>
  );
}

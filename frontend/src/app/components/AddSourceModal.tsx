import { useState } from "react";
import { X, FileText, Youtube, Mic, ClipboardType } from "lucide-react";

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

type SourceType = "document" | "youtube" | "audio" | "text" | null;

export function AddSourceModal({ isOpen, onClose, onAddSource }: AddSourceModalProps) {
  const [selectedType, setSelectedType] = useState<SourceType>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [copiedText, setCopiedText] = useState("");
  const [fileName, setFileName] = useState("");

  if (!isOpen) return null;

  const handleFileUpload = async (
  event: React.ChangeEvent<HTMLInputElement>,
  type: "document" | "audio"
) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const formData = new FormData();
    formData.append("file", file);
    console.log(formData)
    const res = await fetch("http://localhost:8000/api/documents/upload", {
      method: "POST",
      body: formData,
    });
   
    if (!res.ok) {
      throw new Error("Upload failed");
    }

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

  const handleYoutubeSubmit = () => {
    if (youtubeUrl.trim()) {
      const videoId = extractYoutubeId(youtubeUrl);
      onAddSource({
        name: `YouTube: ${videoId || youtubeUrl.substring(0, 30)}...`,
        type: "YouTube Video",
        url: youtubeUrl,
      });
      handleClose();
    }
  };

  const handleTextSubmit = () => {
    if (copiedText.trim()) {
      const preview = copiedText.substring(0, 30);
      onAddSource({
        name: fileName || `Text: ${preview}...`,
        type: "Copied Text",
      });
      handleClose();
    }
  };

  const handleClose = () => {
    setSelectedType(null);
    setYoutubeUrl("");
    setCopiedText("");
    setFileName("");
    onClose();
  };

  const extractYoutubeId = (url: string): string | null => {
    const regExp = /^.*((youtu.be\/)|(v\/)|(\/u\/\w\/)|(embed\/)|(watch\?))\??v?=?([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[7].length === 11 ? match[7] : null;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold">Add Source</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
          >
            <X className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        <div className="p-6">
          {!selectedType ? (
            <div className="space-y-3">
              <button
                onClick={() => document.getElementById("document-upload")?.click()}
                className="w-full flex items-center gap-4 p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all group"
              >
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200">
                  <FileText className="w-6 h-6 text-blue-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-medium">Upload Document</h3>
                  <p className="text-sm text-gray-600">PDF, DOCX, TXT files</p>
                </div>
              </button>
              <input
                id="document-upload"
                type="file"
                accept=".pdf,.doc,.docx,.txt"
                onChange={(e) => handleFileUpload(e, "document")}
                className="hidden"
              />

              <button
                onClick={() => setSelectedType("youtube")}
                className="w-full flex items-center gap-4 p-4 border-2 border-gray-200 rounded-lg hover:border-red-500 hover:bg-red-50 transition-all group"
              >
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center group-hover:bg-red-200">
                  <Youtube className="w-6 h-6 text-red-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-medium">YouTube Link</h3>
                  <p className="text-sm text-gray-600">Add a YouTube video URL</p>
                </div>
              </button>

              <button
                onClick={() => document.getElementById("audio-upload")?.click()}
                className="w-full flex items-center gap-4 p-4 border-2 border-gray-200 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all group"
              >
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center group-hover:bg-purple-200">
                  <Mic className="w-6 h-6 text-purple-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-medium">Audio File</h3>
                  <p className="text-sm text-gray-600">MP3, WAV, M4A files</p>
                </div>
              </button>
              <input
                id="audio-upload"
                type="file"
                accept=".mp3,.wav,.m4a,.aac"
                onChange={(e) => handleFileUpload(e, "audio")}
                className="hidden"
              />

              <button
                onClick={() => setSelectedType("text")}
                className="w-full flex items-center gap-4 p-4 border-2 border-gray-200 rounded-lg hover:border-green-500 hover:bg-green-50 transition-all group"
              >
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center group-hover:bg-green-200">
                  <ClipboardType className="w-6 h-6 text-green-600" />
                </div>
                <div className="text-left">
                  <h3 className="font-medium">Copied Text</h3>
                  <p className="text-sm text-gray-600">Paste text directly</p>
                </div>
              </button>
            </div>
          ) : selectedType === "youtube" ? (
            <div className="space-y-4">
              <button
                onClick={() => setSelectedType(null)}
                className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                ← Back
              </button>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  YouTube URL
                </label>
                <input
                  type="url"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
              </div>
              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => setSelectedType(null)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleYoutubeSubmit}
                  disabled={!youtubeUrl.trim()}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Add
                </button>
              </div>
            </div>
          ) : selectedType === "text" ? (
            <div className="space-y-4">
              <button
                onClick={() => setSelectedType(null)}
                className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                ← Back
              </button>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Name (optional)
                </label>
                <input
                  type="text"
                  value={fileName}
                  onChange={(e) => setFileName(e.target.value)}
                  placeholder="My notes"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Text Content
                </label>
                <textarea
                  value={copiedText}
                  onChange={(e) => setCopiedText(e.target.value)}
                  placeholder="Paste your text here..."
                  rows={6}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  autoFocus
                />
              </div>
              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => setSelectedType(null)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleTextSubmit}
                  disabled={!copiedText.trim()}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Add
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

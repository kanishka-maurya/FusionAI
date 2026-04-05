import { useState } from "react";
import { useNavigate } from "react-router";
import { useAuth } from "../contexts/AuthContext";
import { supabase } from "../contexts/AuthContext";
import { useNotebook } from "../contexts/NotebookContext";
import {
  ArrowLeft,
  Plus,
  BookOpen,
  Clock,
  FileText,
  Sparkles,
} from "lucide-react";

interface Notebook {
  id: string;
  name: string;
  description: string;
  sourcesCount: number;
  messagesCount: number;
  lastModified: string;
  createdAt: string;
}

export function NotebookSessions() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { setNotebook } = useNotebook();
  const [notebooks, setNotebooks] = useState<Notebook[]>([
    {
      id: "1",
      name: "Machine Learning Research",
      description: "Research papers on neural networks and deep learning",
      sourcesCount: 5,
      messagesCount: 24,
      lastModified: "2 hours ago",
      createdAt: "April 3, 2026",
    },
    {
      id: "2",
      name: "AI Ethics Study",
      description: "Documents about AI ethics and responsible AI development",
      sourcesCount: 3,
      messagesCount: 12,
      lastModified: "1 day ago",
      createdAt: "April 1, 2026",
    },
  ]);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newNotebookName, setNewNotebookName] = useState("");
  const [newNotebookDesc, setNewNotebookDesc] = useState("");

  const handleCreateNotebook = async () => {
    if (newNotebookName.trim()) {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      const token = session?.access_token;

      const res = await fetch("http://localhost:8000/api/notebooks/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          name: newNotebookName,
          description: newNotebookDesc,
        }),
      });
      if (!res.ok) {
        alert("Failed to create notebook. Please try again.");
        return;
      }
      if (res.ok) {
        const data = await res.json();
        const newNotebookId = data.notebook_id;
        const { error } = await supabase.from("notebooks").insert([
          {
            notebook_id: newNotebookId,
            name:newNotebookName,
            description: newNotebookDesc,
            user_id: session?.user.id, 
          },
        ]);

        if (error) throw error;
        const newNotebook: Notebook = {
          id: newNotebookId,
          name: newNotebookName,
          description: newNotebookDesc,
          sourcesCount: 0,
          messagesCount: 0,
          lastModified: "Just now",
          createdAt: new Date().toLocaleDateString("en-US", {
            month: "long",
            day: "numeric",
            year: "numeric",
          }),
        };
        setNotebook({ notebook_id: newNotebookId, name: newNotebookName });
        setNotebooks([newNotebook, ...notebooks]);
        setNewNotebookName("");
        setNewNotebookDesc("");
        setShowCreateModal(false);
        navigate(`/notebook/${newNotebook.id}`);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate("/dashboard")}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
                <BookOpen className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  NotebookLM
                </h1>
                <p className="text-sm text-gray-600">Your notebook sessions</p>
              </div>
            </div>
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors">
              {user?.avatar && (
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-9 h-9 rounded-full ring-2 ring-purple-100"
                />
              )}
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  {user?.name}
                </p>
                <p className="text-xs text-gray-500">{user?.email}</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Your Notebooks
            </h2>
            <p className="text-gray-600">
              Create a new notebook or continue with an existing one
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-xl hover:from-purple-700 hover:to-purple-800 transition-all shadow-lg shadow-purple-500/30"
          >
            <Plus className="w-5 h-5" />
            Create New Notebook
          </button>
        </div>

        {/* Notebooks Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {notebooks.map((notebook) => (
            <div
              key={notebook.id}
              onClick={() => navigate(`/notebook/${notebook.id}`)}
              className="bg-white rounded-xl border border-gray-200 hover:border-purple-300 hover:shadow-lg transition-all cursor-pointer group overflow-hidden"
            >
              {/* Card Header */}
              <div className="p-6 border-b border-gray-100 bg-gradient-to-br from-purple-50 to-white group-hover:from-purple-100 transition-colors">
                <div className="flex items-start justify-between mb-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                    <BookOpen className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-xs text-gray-500 bg-white px-3 py-1 rounded-full border border-gray-200">
                    <Clock className="w-3 h-3 inline mr-1" />
                    {notebook.lastModified}
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-purple-700 transition-colors">
                  {notebook.name}
                </h3>
                <p className="text-sm text-gray-600 line-clamp-2">
                  {notebook.description}
                </p>
              </div>

              {/* Card Stats */}
              <div className="p-6">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-gray-600">
                    <FileText className="w-4 h-4" />
                    <span>{notebook.sourcesCount} sources</span>
                  </div>
                  <div className="flex items-center gap-2 text-gray-600">
                    <Sparkles className="w-4 h-4" />
                    <span>{notebook.messagesCount} messages</span>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-xs text-gray-500">
                    Created on {notebook.createdAt}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {/* Empty State - Create First Notebook */}
          {notebooks.length === 0 && (
            <div className="col-span-full flex flex-col items-center justify-center py-20">
              <div className="w-20 h-20 bg-gray-100 rounded-2xl flex items-center justify-center mb-6">
                <BookOpen className="w-10 h-10 text-gray-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                No notebooks yet
              </h3>
              <p className="text-gray-600 mb-6">
                Create your first notebook to get started
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-xl hover:from-purple-700 hover:to-purple-800 transition-all"
              >
                <Plus className="w-5 h-5" />
                Create Notebook
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Create Notebook Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">
              Create New Notebook
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Notebook Name
                </label>
                <input
                  type="text"
                  value={newNotebookName}
                  onChange={(e) => setNewNotebookName(e.target.value)}
                  placeholder="e.g., Machine Learning Research"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description (Optional)
                </label>
                <textarea
                  value={newNotebookDesc}
                  onChange={(e) => setNewNotebookDesc(e.target.value)}
                  placeholder="What is this notebook about?"
                  rows={3}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                />
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewNotebookName("");
                  setNewNotebookDesc("");
                }}
                className="flex-1 px-4 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateNotebook}
                disabled={!newNotebookName.trim()}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create Notebook
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import {
  AlertCircle,
  BookOpen,
  CheckCircle,
  Clock,
  Code,
  ExternalLink,
  FileText,
  Target,
  Video,
} from "lucide-react";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { supabase } from "../contexts/AuthContext";

interface RoadmapNode {
  node_id: string;
  title: string;
  level: string;
  status: string;
  dependencies?: string[] | null;
}

interface NodeContent {
  summary: string;
  estimated_time: string;
  what_you_will_learn: string[];
  topics: Array<{
    title: string;
    explanation: string;
    code_example?: string | null;
    key_takeaway: string;
  }>;
  common_misconceptions: string[];
  resources: Array<{
    type: string;
    title: string;
    url?: string;
  }>;
  practice_questions: Array<{
    question: string;
    hint?: string;
    answer?: string;
  }>;
}

interface NodeContentModalProps {
  node: RoadmapNode;
  roadmapId: string;
  onClose: () => void;
  onStatusChange?: () => void;
}

const API_BASE = "http://localhost:8000";

const normalizeList = (value: any): string[] => {
  if (Array.isArray(value)) return value.filter(Boolean).map(String);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
};

const normalizeContent = (value: any, nodeTitle: string): NodeContent => {
  let parsed = value;

  if (typeof parsed === "string") {
    try {
      parsed = JSON.parse(parsed);
    } catch {
      parsed = { summary: parsed };
    }
  }

  const source = parsed && typeof parsed === "object" ? parsed : {};
  const topics = Array.isArray(source.topics) ? source.topics : [];
  const resources = Array.isArray(source.resources) ? source.resources : [];
  const practiceQuestions = Array.isArray(source.practice_questions)
    ? source.practice_questions
    : [];

  return {
    summary:
      source.summary ||
      `Learn the core concepts behind ${nodeTitle} and how it fits into this roadmap.`,
    estimated_time: source.estimated_time || "1 week",
    what_you_will_learn:
      normalizeList(source.what_you_will_learn).length > 0
        ? normalizeList(source.what_you_will_learn)
        : [
            `Understand ${nodeTitle}`,
            "Connect this topic to the roadmap goal",
            "Practice the concept with focused questions",
          ],
    topics:
      topics.length > 0
        ? topics.map((topic: any, idx: number) => ({
            title: topic?.title || `Concept ${idx + 1}`,
            explanation: topic?.explanation || topic?.summary || String(topic || ""),
            code_example: topic?.code_example ?? null,
            key_takeaway: topic?.key_takeaway || "",
          }))
        : [
            {
              title: nodeTitle,
              explanation:
                source.summary ||
                `This topic introduces the essential ideas needed before moving to dependent roadmap nodes.`,
              code_example: null,
              key_takeaway: `Build a clear mental model of ${nodeTitle}.`,
            },
          ],
    common_misconceptions: normalizeList(source.common_misconceptions),
    resources:
      resources.length > 0
        ? resources.map((resource: any) => ({
            type: resource?.type || "article",
            title: resource?.title || "Learning resource",
            url: resource?.url || "#",
          }))
        : [{ type: "article", title: `Search: ${nodeTitle}`, url: "#" }],
    practice_questions:
      practiceQuestions.length > 0
        ? practiceQuestions.map((question: any, idx: number) => ({
            question: question?.question || String(question || `Practice question ${idx + 1}`),
            hint: question?.hint || "",
            answer: question?.answer || "",
          }))
        : [
            {
              question: `How would you explain ${nodeTitle} in your own words?`,
              hint: "Use one definition, one example, and one reason it matters.",
              answer: "",
            },
          ],
  };
};

export function NodeContentModal({
  node,
  roadmapId,
  onClose,
  onStatusChange,
}: NodeContentModalProps) {
  const [content, setContent] = useState<NodeContent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCompleting, setIsCompleting] = useState(false);
  const [source, setSource] = useState<"cache" | "db" | "generated">("cache");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    loadNodeContent();
  }, [node.node_id]);

  const getToken = async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession();
    return session?.access_token;
  };

  const loadNodeContent = async () => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      const token = await getToken();
      const response = await fetch(
        `${API_BASE}/api/roadmap/${roadmapId}/node/${node.node_id}/content`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Failed to load node content");
      setContent(normalizeContent(data.content, node.title));
      setSource(data.source);
    } catch (error: any) {
      console.error("Failed to load node content:", error);
      setErrorMessage(error.message || "Failed to load content");
    } finally {
      setIsLoading(false);
    }
  };

  const markCompleted = async () => {
    setIsCompleting(true);
    setErrorMessage("");
    try {
      const token = await getToken();
      const response = await fetch(
        `${API_BASE}/api/roadmap/${roadmapId}/node/${node.node_id}/status?status=done`,
        {
          method: "PATCH",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Failed to update node status");
      onStatusChange?.();
      onClose();
    } catch (error: any) {
      console.error("Failed to update node status:", error);
      setErrorMessage(error.message || "Failed to update status");
    } finally {
      setIsCompleting(false);
    }
  };

  const getResourceIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case "video":
        return <Video className="w-4 h-4" />;
      case "article":
      case "paper":
        return <FileText className="w-4 h-4" />;
      case "code":
      case "tutorial":
        return <Code className="w-4 h-4" />;
      default:
        return <BookOpen className="w-4 h-4" />;
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-[#111322] border border-white/10 text-white">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider">
            <BookOpen className="w-5 h-5 text-blue-400" />
            {node.title}
          </DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="py-12 text-center">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Generating...
            </p>
          </div>
        ) : content ? (
          <div className="space-y-6">
            {errorMessage && (
              <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
                {errorMessage}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                <Clock className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="text-[10px] text-blue-300 font-bold uppercase tracking-wider">Time</p>
                  <p className="text-sm text-white font-semibold">{content.estimated_time}</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-4 bg-violet-500/10 border border-violet-500/20 rounded-xl">
                <Target className="w-5 h-5 text-violet-400" />
                <div>
                  <p className="text-[10px] text-violet-300 font-bold uppercase tracking-wider">Level</p>
                  <p className="text-sm text-white font-semibold capitalize">{node.level}</p>
                </div>
              </div>
            </div>

            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4 bg-black/30 border border-white/5">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="topics">Topics</TabsTrigger>
                <TabsTrigger value="resources">Resources</TabsTrigger>
                <TabsTrigger value="practice">Practice</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-5 mt-5">
                <section>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Summary</h3>
                  <p className="text-sm text-slate-200 leading-relaxed">{content.summary}</p>
                </section>

                <section>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                    Outcomes
                  </h3>
                  <ul className="space-y-2">
                    {content.what_you_will_learn.map((objective, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </section>

                {!!content.common_misconceptions?.length && (
                  <section>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                      Common Misconceptions
                    </h3>
                    <ul className="space-y-2">
                      {content.common_misconceptions.map((item, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
              </TabsContent>

              <TabsContent value="topics" className="space-y-4 mt-5">
                {content.topics.map((topic, idx) => (
                  <div key={idx} className="p-4 bg-black/20 border border-white/5 rounded-xl">
                    <h4 className="font-bold text-white mb-2">{topic.title}</h4>
                    <p className="text-sm text-slate-300 leading-relaxed">{topic.explanation}</p>
                    {topic.code_example && (
                      <pre className="mt-3 p-3 rounded-lg bg-black/40 border border-white/5 overflow-x-auto text-xs text-cyan-200">
                        <code>{topic.code_example}</code>
                      </pre>
                    )}
                    <p className="mt-3 text-xs text-blue-300 font-semibold">
                      {topic.key_takeaway}
                    </p>
                  </div>
                ))}
              </TabsContent>

              <TabsContent value="resources" className="space-y-3 mt-5">
                {content.resources.map((resource, idx) => (
                  <a
                    key={idx}
                    href={resource.url || "#"}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-start gap-3 p-4 bg-black/20 border border-white/5 rounded-xl hover:border-blue-500/30 transition-all"
                  >
                    <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center shrink-0 text-blue-300">
                      {getResourceIcon(resource.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-1 gap-3">
                        <h4 className="font-semibold text-white">{resource.title}</h4>
                        <ExternalLink className="w-4 h-4 text-slate-500 shrink-0" />
                      </div>
                      <span className="inline-block px-2 py-1 bg-white/5 text-slate-300 text-[10px] rounded-lg uppercase tracking-wider">
                        {resource.type}
                      </span>
                    </div>
                  </a>
                ))}
              </TabsContent>

              <TabsContent value="practice" className="space-y-3 mt-5">
                {content.practice_questions.map((question, idx) => (
                  <div key={idx} className="p-4 bg-black/20 border border-white/5 rounded-xl">
                    <h4 className="font-semibold text-white mb-2">{question.question}</h4>
                    {question.hint && (
                      <p className="text-xs text-amber-300 mb-2">Hint: {question.hint}</p>
                    )}
                    {question.answer && (
                      <p className="text-sm text-slate-300 leading-relaxed">{question.answer}</p>
                    )}
                  </div>
                ))}
              </TabsContent>
            </Tabs>

            <div className="flex gap-3 pt-4 border-t border-white/5">
              <Button
                variant="outline"
                className="flex-1 bg-transparent border-white/10 text-slate-300 hover:bg-white/5 hover:text-white"
                onClick={onClose}
              >
                Close
              </Button>
              <Button
                className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                onClick={markCompleted}
                disabled={isCompleting}
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                {isCompleting ? "Updating..." : "Mark Done"}
              </Button>
            </div>

          </div>
        ) : (
          <div className="py-12 text-center">
            <AlertCircle className="w-12 h-12 text-rose-400 mx-auto mb-4" />
            <p className="text-sm text-slate-400">{errorMessage || "Failed to load content"}</p>
            <Button variant="outline" className="mt-4" onClick={loadNodeContent}>
              Retry
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

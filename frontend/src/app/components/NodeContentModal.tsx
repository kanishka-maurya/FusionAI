import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import {
  BookOpen,
  Clock,
  Target,
  CheckCircle,
  AlertCircle,
  FileText,
  Video,
  Code,
  ExternalLink,
} from "lucide-react";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

interface RoadmapNode {
  id: string;
  title: string;
  description: string;
  level: string;
  status: "locked" | "unlocked" | "completed";
  dependencies: string[];
  content_generated: boolean;
}

interface NodeContent {
  overview: string;
  learning_objectives: string[];
  key_concepts: Array<{
    title: string;
    explanation: string;
  }>;
  resources: Array<{
    type: string;
    title: string;
    url?: string;
    description: string;
  }>;
  practice_exercises: Array<{
    title: string;
    difficulty: string;
    description: string;
  }>;
  estimated_time: string;
  prerequisites_summary: string[];
}

interface NodeContentModalProps {
  node: RoadmapNode;
  roadmapId: string;
  onClose: () => void;
}

export function NodeContentModal({ node, roadmapId, onClose }: NodeContentModalProps) {
  const [content, setContent] = useState<NodeContent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [source, setSource] = useState<"cache" | "db" | "generated">("cache");

  useEffect(() => {
    loadNodeContent();
  }, [node.id]);

  const loadNodeContent = async () => {
    setIsLoading(true);
    try {
      // TODO: Replace with actual API call
      // const response = await fetch(`/api/roadmap/${roadmapId}/node/${node.id}/content`);
      // const data = await response.json();
      // setContent(data.content);
      // setSource(data.source);

      // Mock data
      await new Promise(resolve => setTimeout(resolve, 1000));

      const mockContent: NodeContent = {
        overview: `This section covers ${node.title.toLowerCase()}, providing you with essential knowledge and practical skills. You'll learn through a combination of theory, examples, and hands-on exercises designed for ${node.level}-level learners.`,
        learning_objectives: [
          `Understand the core principles of ${node.title.toLowerCase()}`,
          "Apply concepts through practical examples",
          "Build a solid foundation for advanced topics",
          "Develop problem-solving skills in this area",
        ],
        key_concepts: [
          {
            title: "Fundamental Principles",
            explanation: "Core theoretical foundations and why they matter in real-world applications.",
          },
          {
            title: "Practical Applications",
            explanation: "How these concepts are used in industry and common use cases.",
          },
          {
            title: "Best Practices",
            explanation: "Recommended approaches and patterns for implementing these concepts.",
          },
          {
            title: "Common Pitfalls",
            explanation: "Mistakes to avoid and how to troubleshoot common issues.",
          },
        ],
        resources: [
          {
            type: "article",
            title: `Introduction to ${node.title}`,
            url: "#",
            description: "A comprehensive guide covering all the basics you need to know.",
          },
          {
            type: "video",
            title: `${node.title} Explained`,
            url: "#",
            description: "Visual walkthrough with practical examples and demonstrations.",
          },
          {
            type: "documentation",
            title: "Official Documentation",
            url: "#",
            description: "Reference material and detailed technical specifications.",
          },
          {
            type: "tutorial",
            title: "Hands-on Tutorial",
            url: "#",
            description: "Step-by-step guide to building your first project.",
          },
        ],
        practice_exercises: [
          {
            title: "Basic Exercise",
            difficulty: "Easy",
            description: "Practice fundamental concepts with guided examples.",
          },
          {
            title: "Intermediate Challenge",
            difficulty: "Medium",
            description: "Apply your knowledge to solve real-world problems.",
          },
          {
            title: "Advanced Project",
            difficulty: "Hard",
            description: "Build a complete solution demonstrating mastery.",
          },
        ],
        estimated_time: "4-6 hours",
        prerequisites_summary: node.dependencies.length > 0
          ? ["Complete previous topics in the roadmap", "Basic understanding of prerequisite concepts"]
          : ["No prerequisites required"],
      };

      setContent(mockContent);
      setSource("generated");
    } catch (error) {
      console.error("Failed to load node content:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const getResourceIcon = (type: string) => {
    switch (type) {
      case "video":
        return <Video className="w-4 h-4" />;
      case "article":
        return <FileText className="w-4 h-4" />;
      case "documentation":
        return <BookOpen className="w-4 h-4" />;
      case "tutorial":
        return <Code className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case "easy":
        return "bg-green-100 text-green-700";
      case "medium":
        return "bg-yellow-100 text-yellow-700";
      case "hard":
        return "bg-red-100 text-red-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl">
            <BookOpen className="w-6 h-6 text-purple-600" />
            {node.title}
          </DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="py-12 text-center">
            <div className="w-12 h-12 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-600">Loading content...</p>
          </div>
        ) : content ? (
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg">
                <Clock className="w-5 h-5 text-blue-600" />
                <div>
                  <p className="text-xs text-blue-600 font-medium">Estimated Time</p>
                  <p className="text-sm text-blue-900 font-semibold">{content.estimated_time}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 p-3 bg-purple-50 rounded-lg">
                <Target className="w-5 h-5 text-purple-600" />
                <div>
                  <p className="text-xs text-purple-600 font-medium">Level</p>
                  <p className="text-sm text-purple-900 font-semibold capitalize">{node.level}</p>
                </div>
              </div>
            </div>

            {/* Tabs for Content */}
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="concepts">Concepts</TabsTrigger>
                <TabsTrigger value="resources">Resources</TabsTrigger>
                <TabsTrigger value="practice">Practice</TabsTrigger>
              </TabsList>

              {/* Overview Tab */}
              <TabsContent value="overview" className="space-y-4 mt-4">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-2">Overview</h3>
                  <p className="text-gray-700 leading-relaxed">{content.overview}</p>
                </div>

                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Learning Objectives</h3>
                  <ul className="space-y-2">
                    {content.learning_objectives.map((objective, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <span className="text-gray-700">{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {content.prerequisites_summary.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3">Prerequisites</h3>
                    <ul className="space-y-2">
                      {content.prerequisites_summary.map((prereq, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <AlertCircle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700">{prereq}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </TabsContent>

              {/* Key Concepts Tab */}
              <TabsContent value="concepts" className="space-y-4 mt-4">
                {content.key_concepts.map((concept, idx) => (
                  <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <h4 className="font-semibold text-gray-900 mb-2">{concept.title}</h4>
                    <p className="text-gray-700 text-sm">{concept.explanation}</p>
                  </div>
                ))}
              </TabsContent>

              {/* Resources Tab */}
              <TabsContent value="resources" className="space-y-3 mt-4">
                {content.resources.map((resource, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-4 bg-white border border-gray-200 rounded-lg hover:border-purple-300 hover:shadow-md transition-all cursor-pointer"
                  >
                    <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      {getResourceIcon(resource.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-1">
                        <h4 className="font-semibold text-gray-900">{resource.title}</h4>
                        <ExternalLink className="w-4 h-4 text-gray-400" />
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{resource.description}</p>
                      <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full capitalize">
                        {resource.type}
                      </span>
                    </div>
                  </div>
                ))}
              </TabsContent>

              {/* Practice Tab */}
              <TabsContent value="practice" className="space-y-3 mt-4">
                {content.practice_exercises.map((exercise, idx) => (
                  <div key={idx} className="p-4 bg-white border border-gray-200 rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-semibold text-gray-900">{exercise.title}</h4>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(exercise.difficulty)}`}>
                        {exercise.difficulty}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{exercise.description}</p>
                  </div>
                ))}
              </TabsContent>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4 border-t">
              <Button
                variant="outline"
                className="flex-1"
                onClick={onClose}
              >
                Close
              </Button>
              <Button
                className="flex-1 bg-green-600 hover:bg-green-700"
                onClick={() => {
                  // TODO: Mark node as completed
                  console.log("Mark as completed");
                }}
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                Mark as Completed
              </Button>
            </div>

            {/* Debug Info */}
            {source && (
              <div className="text-xs text-gray-500 text-center pt-2 border-t">
                Content source: {source}
              </div>
            )}
          </div>
        ) : (
          <div className="py-12 text-center">
            <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
            <p className="text-gray-600">Failed to load content</p>
            <Button variant="outline" className="mt-4" onClick={loadNodeContent}>
              Retry
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

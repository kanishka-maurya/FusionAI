import { MessageSquare, Lightbulb, FileSearch, Sparkles } from "lucide-react";

export function EmptyState() {
  const suggestions = [
    {
      icon: FileSearch,
      text: "Summarize the main points from my documents",
    },
    {
      icon: Lightbulb,
      text: "What are the key insights I should know?",
    },
    {
      icon: Sparkles,
      text: "Compare and contrast different perspectives",
    },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-8">
        <div>
          <MessageSquare className="w-16 h-16 mx-auto mb-4 text-blue-600" />
          <h1 className="text-3xl font-semibold mb-2">
            Ask anything from our Chatbot !
          </h1>
          <p className="text-gray-600">
            Upload your sources and start asking questions. I'll help you understand, analyze, and explore your documents.
          </p>
        </div>

        <div className="space-y-3">
          <p className="text-sm font-medium text-gray-700">Try asking:</p>
          <div className="grid gap-3">
            {suggestions.map((suggestion, index) => {
              const Icon = suggestion.icon;
              return (
                <div
                  key={index}
                  className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg text-left hover:bg-gray-100 transition-colors cursor-pointer"
                >
                  <Icon className="w-5 h-5 text-blue-600 flex-shrink-0" />
                  <span className="text-sm text-gray-700">{suggestion.text}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

import { createContext, useContext, useState, ReactNode, useEffect } from "react";

interface Notebook {
  notebook_id: string;
  name: string;
}

interface NotebookContextType {
  currentNotebook: Notebook | null;
  setNotebook: (notebook: Notebook) => void;
  clearNotebook: () => void;
}

const NotebookContext = createContext<NotebookContextType | undefined>(undefined);

export function NotebookProvider({ children }: { children: ReactNode }) {
  const [currentNotebook, setCurrentNotebook] = useState<Notebook | null>(() => {
    const saved = localStorage.getItem("active_notebook");
    return saved ? JSON.parse(saved) : null;
  });

  const setNotebook = (notebook: Notebook) => {
    setCurrentNotebook(notebook);
    localStorage.setItem("active_notebook", JSON.stringify(notebook));
  };

  const clearNotebook = () => {
    setCurrentNotebook(null);
    localStorage.removeItem("active_notebook");
  };

  return (
    <NotebookContext.Provider value={{ currentNotebook, setNotebook, clearNotebook }}>
      {children}
    </NotebookContext.Provider>
  );
}

export function useNotebook() {
  const context = useContext(NotebookContext);
  if (context === undefined) {
    throw new Error("useNotebook must be used within a NotebookProvider");
  }
  return context;
}

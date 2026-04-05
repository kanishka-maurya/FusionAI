import { createBrowserRouter, Navigate, Outlet } from "react-router-dom"; 
import { AuthPage } from "./components/AuthPage";
import { Dashboard } from "./components/Dashboard";
import { NotebookSessions } from "./components/NotebookSessions";
import MainLayout from "./components/MainLayout";
import { RoadmapPage } from "./components/RoadmapPage";
import { AINewsPage } from "./components/AINewsPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { NotebookProvider } from "./contexts/NotebookContext";

export const router = createBrowserRouter([
  {
    path: "/auth",
    element: <AuthPage />,
  },
  {
    element: (
      <ProtectedRoute>
        <NotebookProvider>
          {/* Outlet renders whichever child route is active */}
          <Outlet /> 
        </NotebookProvider>
      </ProtectedRoute>
    ),
    children: [
      {
        path: "/",
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: "/dashboard",
        element: <Dashboard />,
      },
      {
        path: "/notebook",
        element: <NotebookSessions />,
      },
      {
        path: "/notebook/:notebookId",
        element: <MainLayout />,
      },
      {
        path: "/roadmap",
        element: <RoadmapPage />,
      },
      {
        path: "/ai-news",
        element: <AINewsPage />,
      },
    ],
  },
  {
    path: "*",
    element: <Navigate to="/dashboard" replace />,
  },
]);

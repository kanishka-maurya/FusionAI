import { createBrowserRouter, Navigate } from "react-router";
import { AuthPage } from "./components/AuthPage";
import  {Dashboard} from "./components/Dashboard";
import { NotebookSessions } from "./components/NotebookSessions";
import MainLayout  from "./components/MainLayout";
import { RoadmapPage } from "./components/RoadmapPage";
import { AINewsPage } from "./components/AINewsPage";
import { ProtectedRoute } from "./components/ProtectedRoute";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: "/dashboard",
    element: (
      <ProtectedRoute>
        <Dashboard />
      </ProtectedRoute>
    ),
  },
  {
    path: "/notebook",
    element: (
      <ProtectedRoute>
        <NotebookSessions />
      </ProtectedRoute>
    ),
  },
  {
    path: "/notebook/:notebookId",
    element: (
      <ProtectedRoute>
        <MainLayout />
      </ProtectedRoute>
    ),
  },
  {
    path: "/roadmap",
    element: (
      <ProtectedRoute>
        <RoadmapPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/ai-news",
    element: (
      <ProtectedRoute>
        <AINewsPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/auth",
    element: <AuthPage />,
  },
  {
    path: "*",
    element: <Navigate to="/dashboard" replace />,
  },
]);

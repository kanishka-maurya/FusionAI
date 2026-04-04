import { createBrowserRouter, Navigate } from "react-router";
import { AuthPage } from "./components/AuthPage";
import MainLayout from "./components/MainLayout";
import { ProtectedRoute } from "./components/ProtectedRoute";

export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <ProtectedRoute>
        <MainLayout />
      </ProtectedRoute>
    ),
  },
  {
    path: "/auth",
    element: <AuthPage />,
  },
  {
    path: "*",
    element: <Navigate to="/" replace />,
  },
]);

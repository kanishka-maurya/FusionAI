import { useState } from "react";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

export function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const { login, loginWithGoogle, signup } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await signup(name, email, password);
      }
      navigate("/");
    } catch (err: any) {
      setError(err.message || "Authentication failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleAuth = async () => {
    setError("");
    setIsLoading(true);

    try {
      await loginWithGoogle();
    } catch (err: any) {
      setError(err.message || "Google authentication failed.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 px-4">
      <div className="w-full max-w-md">
        {/* Transparent Glass Container */}
        <div className="bg-slate-900/40 border border-white/10 rounded-2xl shadow-2xl p-8 backdrop-blur-md">
          
          {/* Logo/Header */}
          <div className="text-center mb-8">
            <div
              className="inline-flex items-center justify-center w-20 h-20 rounded-3xl mb-4 shadow-xl relative overflow-hidden"
              style={{
                background:
                  "linear-gradient(135deg, #0891b2 0%, #2563eb 40%, #7e22ce 100%)",
              }}
            >
              {/* Prism Glow */}
              <div
                className="absolute inset-0 opacity-40 blur-xl"
                style={{
                  background:
                    "radial-gradient(circle at center, #0891b2 0%, #2563eb 70%, #7e22ce 100%)",
                }}
              />

              {/* Knowledge Prism Logo */}
              <svg
                className="w-12 h-12 relative z-10"
                viewBox="0 0 100 100"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Prism Core */}
                <polygon
                  points="50,22 78,70 22,70"
                  fill="url(#prismGradient)"
                  stroke="#7e22ce"
                  strokeWidth="2"
                  opacity="0.95"
                />

                {/* Left Ray */}
                <path
                  d="M10 54 L32 58"
                  stroke="#f1a598"
                  strokeWidth="4"
                  strokeLinecap="round"
                />

                {/* Center Ray */}
                <path
                  d="M50 8 L50 30"
                  stroke="#ffffff"
                  strokeWidth="4"
                  strokeLinecap="round"
                />

                {/* Right Ray */}
                <path
                  d="M90 54 L68 58"
                  stroke="#a2c0e2"
                  strokeWidth="4"
                  strokeLinecap="round"
                />

                {/* Internal AI Connection */}
                <circle cx="50" cy="50" r="4" fill="#ffffff" />
                <path
                  d="M50 50 L36 62"
                  stroke="#ffffff"
                  strokeWidth="2"
                  opacity="0.8"
                />
                <path
                  d="M50 50 L64 62"
                  stroke="#ffffff"
                  strokeWidth="2"
                  opacity="0.8"
                />

                {/* Gradient */}
                <defs>
                  <linearGradient
                    id="prismGradient"
                    x1="0"
                    y1="0"
                    x2="100"
                    y2="100"
                  >
                    <stop offset="0%" stopColor="#f1a598" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#a2c0e2" stopOpacity="0.5" />
                  </linearGradient>
                </defs>
              </svg>
            </div>

            {/* Platform Name */}
            <h1 className="text-3xl font-extrabold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
              FusionAI
            </h1>
            <p className="text-xs text-slate-400 font-medium mt-1">
              {isLogin ? "Sign in to your account" : "Create your master profile"}
            </p>
          </div>

          {/* Google Sign In Button */}
          <button
            type="button"
            onClick={handleGoogleAuth}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-slate-200 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed mb-6 shadow-sm"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            <span className="font-semibold text-xs tracking-wide">
              Continue with Google
            </span>
          </button>

          {/* Divider */}
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/10"></div>
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="px-4 bg-slate-950/80 rounded-full text-slate-400 font-medium border border-white/5">
                Or continue with email
              </span>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-950/40 border border-red-500/30 rounded-xl text-red-400 text-xs font-medium backdrop-blur-sm">
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label
                  htmlFor="name"
                  className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5"
                >
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required={!isLogin}
                  className="w-full px-4 py-3 bg-slate-950/40 border border-white/10 rounded-xl text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50 focus:border-transparent outline-none transition placeholder-slate-600"
                  placeholder="John Doe"
                />
              </div>
            )}

            <div>
              <label
                htmlFor="email"
                className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5"
              >
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 bg-slate-950/40 border border-white/10 rounded-xl text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50 focus:border-transparent outline-none transition placeholder-slate-600"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-1.5"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-3 bg-slate-950/40 border border-white/10 rounded-xl text-slate-200 text-sm focus:ring-2 focus:ring-cyan-500/50 focus:border-transparent outline-none transition placeholder-slate-600"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-xl font-semibold text-xs uppercase tracking-wider hover:from-cyan-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md mt-2"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  {isLogin ? "Signing in..." : "Creating account..."}
                </span>
              ) : isLogin ? (
                "Sign In"
              ) : (
                "Create Account"
              )}
            </button>
          </form>

          {/* Toggle Login/Signup */}
          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => {
                setIsLogin(!isLogin);
                setError("");
              }}
              className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              {isLogin
                ? "Don't have an account? Sign up"
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
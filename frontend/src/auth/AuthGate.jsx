import React, { useEffect } from 'react';
import { useAuth } from './AuthContext';

export default function AuthGate({ children }) {
  const { configured, loading, session } = useAuth();
  const onLogin = window.location.pathname === '/login'
    || new URLSearchParams(window.location.search).get('page') === 'login';

  useEffect(() => {
    if (configured && !loading && !session && !onLogin) window.location.replace('/login');
    if (configured && !loading && session && onLogin) window.location.replace('/');
  }, [configured, loading, session, onLogin]);

  if (configured && loading) {
    return <main className="auth-loading" role="status">Checking your sign-in…</main>;
  }
  if (configured && ((!session && !onLogin) || (session && onLogin))) return null;
  return children;
}

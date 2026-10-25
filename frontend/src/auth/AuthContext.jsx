import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { supabase, supabaseConfigured } from './supabase';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(supabaseConfigured);

  useEffect(() => {
    if (!supabase) return undefined;
    let active = true;
    supabase.auth.getSession().then(({ data, error }) => {
      if (!active) return;
      if (error) console.error('Unable to restore the sign-in session.', error);
      setSession(data?.session ?? null);
      setLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setLoading(false);
    });
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, []);

  const value = useMemo(() => ({
    configured: supabaseConfigured,
    loading,
    session,
    user: session?.user ?? null,
    async requestSignInCode(email) {
      if (!supabase) throw new Error('Supabase sign-in is not configured.');
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { shouldCreateUser: false },
      });
      if (error) throw error;
    },
    async verifySignInCode(email, token) {
      if (!supabase) throw new Error('Supabase sign-in is not configured.');
      const { error } = await supabase.auth.verifyOtp({ email, token, type: 'email' });
      if (error) throw error;
    },
    async signOut() {
      if (!supabase) return;
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
    },
  }), [loading, session]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}

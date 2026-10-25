import React, { useState } from 'react';
import { Activity, ArrowLeft, ArrowRight, Eye, EyeOff, LockKeyhole, UserRound, Moon, Sun } from 'lucide-react';
import { useTheme } from './state/ThemeContext';
import { useAuth } from './auth/AuthContext';
import brandLogo from './assets/q-diagnose-logo.png';

export default function LoginPage() {
  const { theme, toggleTheme } = useTheme();
  const { configured: authConfigured, signIn } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [notice, setNotice] = useState('');
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!authConfigured) {
      setNotice('Authentication is not configured. This app is available only as a local preview.');
      return;
    }
    setSubmitting(true);
    setNotice('');
    const form = new FormData(event.currentTarget);
    try {
      await signIn(String(form.get('email') || ''), String(form.get('password') || ''));
      window.location.replace('/');
    } catch (error) {
      setNotice(error.message || 'Sign-in failed. Check your email and password.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-shell login-glass-shell">
      <div className="login-glass-panel">
      <section className="login-story login-glass-story" aria-label="Q-Diagnose brand">
        <a className="login-brand login-brand-centered" href="/" aria-label="Q-Diagnose home">
          <img className="login-brand-logo" src={brandLogo} alt="Q-Diagnose — Hybrid Quantum Intelligence for Better Diagnostics" />
        </a>
      </section>

      <section className="login-form-side login-glass-side" aria-labelledby="login-title">
        <button className="login-theme-toggle" type="button" onClick={toggleTheme} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'bright'} theme`}>
          {theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}
        </button>

        <div className="login-card login-glass-card">
          <div className="login-heading">
            <p className="login-kicker">WELCOME BACK</p>
            <h2 id="login-title">Sign in</h2>
          </div>

          <form className="login-form" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="login-email">Email address</label>
            <div className="login-input-wrap"><UserRound size={16} aria-hidden="true" /><input id="login-email" name="email" type="email" autoComplete="username" placeholder="Email address" required /></div>

            <label className="sr-only" htmlFor="login-password">Password</label>
            <div className="login-password-wrap">
              <LockKeyhole className="login-password-icon" size={16} aria-hidden="true" />
              <input id="login-password" name="password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" placeholder="Enter your password" required />
              <button type="button" aria-label={showPassword ? 'Hide password' : 'Show password'} onClick={() => setShowPassword(value => !value)}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button>
            </div>

            <div className="login-options login-options-end"><a href="#password-help" onClick={(event) => { event.preventDefault(); setNotice(authConfigured ? 'Contact your administrator to reset your account password.' : 'Password recovery is unavailable because authentication is not configured in this local build.'); }}>Forgot password?</a></div>
            <button className="login-submit" type="submit" disabled={submitting || !authConfigured}>{submitting ? 'Signing in…' : 'Sign in'} <ArrowRight size={17} aria-hidden="true" /></button>
            {notice && <p className="login-notice" role="status">{notice}</p>}
          </form>
          <p className="login-disclaimer">{authConfigured ? 'Sign-in protected by Supabase Auth.' : 'Local preview · Authentication is not configured.'}</p>
        </div>
      </section>
      </div>
    </main>
  );
}

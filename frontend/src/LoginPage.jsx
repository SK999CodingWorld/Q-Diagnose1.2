import React, { useState } from 'react';
import { ArrowLeft, ArrowRight, KeyRound, Mail, Moon, Sun, UserRound } from 'lucide-react';
import { useTheme } from './state/ThemeContext';
import { useAuth } from './auth/AuthContext';
import brandLogo from './assets/q-diagnose-logo.png';

export default function LoginPage() {
  const { theme, toggleTheme } = useTheme();
  const { configured: authConfigured, requestSignInCode, verifySignInCode } = useAuth();
  const [step, setStep] = useState('email');
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
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
    try {
      if (step === 'email') {
        await requestSignInCode(email.trim());
        setStep('code');
        setNotice('A one-time sign-in code has been sent. Check your inbox and spam folder.');
      } else {
        await verifySignInCode(email.trim(), code.trim());
        window.location.replace('/');
      }
    } catch (error) {
      setNotice(error.message || (step === 'email' ? 'Could not send a sign-in code. Try again shortly.' : 'That code could not be verified. Check it and try again.'));
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
            <p className="login-kicker">SECURE EMAIL SIGN-IN</p>
            <h2 id="login-title">Sign in with a code</h2>
          </div>

          <form className="login-form" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="login-email">Email address</label>
            <div className="login-input-wrap"><UserRound size={16} aria-hidden="true" /><input id="login-email" name="email" type="email" autoComplete="email" placeholder="Email address" value={email} onChange={event => setEmail(event.target.value)} required disabled={step === 'code'} /></div>

            {step === 'code' && <>
              <label className="sr-only" htmlFor="login-code">Email verification code</label>
              <div className="login-input-wrap"><KeyRound size={16} aria-hidden="true" /><input id="login-code" name="code" type="text" inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} placeholder="6-digit email code" value={code} onChange={event => setCode(event.target.value.replace(/\D/g, '').slice(0, 6))} required /></div>
              <div className="login-options">
                <button className="login-back-link" type="button" onClick={() => { setStep('email'); setCode(''); setNotice(''); }}><ArrowLeft size={14} aria-hidden="true" /> Use a different email</button>
                <button className="login-back-link" type="button" disabled={submitting} onClick={async () => {
                  setSubmitting(true);
                  setNotice('');
                  try {
                    await requestSignInCode(email.trim());
                    setNotice('A new sign-in code has been sent. Check your inbox and spam folder.');
                  } catch (error) {
                    setNotice(error.message || 'Could not resend the code. Wait a moment and try again.');
                  } finally {
                    setSubmitting(false);
                  }
                }}>Send another code</button>
              </div>
            </>}

            <button className="login-submit" type="submit" disabled={submitting || !authConfigured}>{submitting ? (step === 'email' ? 'Sending code…' : 'Verifying…') : (step === 'email' ? 'Email me a sign-in code' : 'Verify code & sign in')} {step === 'email' ? <Mail size={17} aria-hidden="true" /> : <ArrowRight size={17} aria-hidden="true" />}</button>
            {notice && <p className="login-notice" role="status">{notice}</p>}
          </form>
          <p className="login-disclaimer">{authConfigured ? 'We email a one-time verification code, never a reusable password.' : 'Local preview · Authentication is not configured.'}</p>
        </div>
      </section>
      </div>
    </main>
  );
}

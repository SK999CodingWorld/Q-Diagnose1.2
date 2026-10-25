import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error('Dashboard render failure', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="mx-auto flex min-h-screen max-w-xl flex-col items-start justify-center gap-4 px-6 text-slate-100">
          <h1 className="text-2xl font-semibold">The dashboard could not load</h1>
          <p className="text-sm text-slate-400">Reload the page. If the problem continues, check the browser console or contact your project administrator.</p>
          <button type="button" onClick={() => window.location.reload()} className="rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300">Reload dashboard</button>
        </main>
      );
    }
    return this.props.children;
  }
}

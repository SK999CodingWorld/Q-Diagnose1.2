import React from 'react';
import { Database, Moon, Sun, UploadCloud, LibraryBig, LogIn, LogOut } from 'lucide-react';
import { useTheme } from '../state/ThemeContext';
import { useAuth } from '../auth/AuthContext';
import brandLogo from '../assets/q-diagnose-logo.png';

export default function AppHeader({ datasets, selectedDataset, onDatasetChange, onUpload, onBrowseDatasets, showResearchDatasetControls = false }) {
  const { theme, toggleTheme } = useTheme();
  const { configured: authConfigured, session, signOut } = useAuth();
  const nextTheme = theme === 'light' ? 'dark' : 'light';
  async function handleSignOut() {
    try {
      await signOut();
      window.location.replace('/login');
    } catch (error) {
      window.dispatchEvent(new CustomEvent('qdiagnose:api-error', { detail: { message: error.message } }));
    }
  }
  return (
    <header className="app-header sticky top-0 z-40 border-b px-4 py-3 backdrop-blur-xl sm:px-7">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <img
            className="app-header-logo h-16 w-16 shrink-0 object-contain sm:h-[4.5rem] sm:w-[4.5rem]"
            src={brandLogo}
            alt="Q-Diagnose logo"
          />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="app-brand-title text-lg font-bold tracking-tight">Q-Diagnose</h1>
              <span className={`app-mode-badge rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide ${showResearchDatasetControls ? 'is-research' : ''}`}>{showResearchDatasetControls ? 'Model research' : 'Lab report review'}</span>
            </div>
            <p className="app-header-description hidden text-xs sm:block">{showResearchDatasetControls ? 'Cohort experiments and model evaluation' : 'Upload and review laboratory results'}</p>
          </div>
        </div>

        <div className="flex w-full items-center gap-2 sm:w-auto">
          {showResearchDatasetControls && <>
          <label className="app-header-control flex min-w-0 flex-1 items-center gap-2 rounded-xl border px-3 py-2.5 transition sm:flex-initial">
            <Database className="h-4 w-4 shrink-0" aria-hidden="true" />
            <span className="sr-only">Active dataset</span>
            <select
              aria-label="Active dataset"
              className="app-header-select w-full min-w-0 flex-1 cursor-pointer truncate bg-transparent pr-2 text-sm font-medium focus:outline-none sm:max-w-64"
              value={selectedDataset}
              onChange={event => onDatasetChange(event.target.value)}
              disabled={datasets.length === 0}
            >
              {datasets.length === 0 && <option value="">No datasets available</option>}
              {datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.n_rows} rows</option>)}
            </select>
          </label>
          <button type="button" onClick={onUpload} aria-label="Upload research cohort" title="Upload research cohort" className="app-header-primary inline-flex shrink-0 items-center justify-center gap-2 rounded-xl px-3.5 py-2.5 text-sm font-semibold shadow-sm transition sm:px-3.5 md:px-4">
            <UploadCloud className="h-4 w-4" aria-hidden="true" />
            <span className="hidden md:inline">Upload cohort</span>
          </button>
          <button type="button" onClick={onBrowseDatasets} aria-label="Browse public datasets" title="Browse public datasets" className="app-header-secondary inline-flex shrink-0 items-center justify-center gap-2 rounded-xl border px-3 py-2.5 text-sm font-semibold">
            <LibraryBig className="h-4 w-4" aria-hidden="true" />
            <span className="hidden lg:inline">Public datasets</span>
          </button>
          </>}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${nextTheme === 'light' ? 'bright' : 'dark'} theme`}
            title={`Switch to ${nextTheme === 'light' ? 'bright' : 'dark'} theme`}
            aria-pressed={theme === 'dark'}
            className="theme-toggle app-header-icon-button inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border shadow-sm"
          >
            {theme === 'light' ? <Moon className="h-4 w-4" aria-hidden="true" /> : <Sun className="h-4 w-4" aria-hidden="true" />}
          </button>
          {authConfigured && session
            ? <button type="button" onClick={handleSignOut} aria-label="Sign out" title="Sign out" className="app-login-link inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-xl border px-3 text-sm font-semibold shadow-sm transition"><LogOut className="h-4 w-4" aria-hidden="true" /><span className="hidden sm:inline">Sign out</span></button>
            : <a href="/login" aria-label="Sign in" title="Sign in" className="app-login-link inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-xl border px-3 text-sm font-semibold shadow-sm transition"><LogIn className="h-4 w-4" aria-hidden="true" /><span className="hidden sm:inline">Sign in</span></a>}
        </div>
      </div>
    </header>
  );
}

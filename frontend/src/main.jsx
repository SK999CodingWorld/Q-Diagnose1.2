import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import LoginPage from './LoginPage.jsx'
import ApiErrorNotice from './components/ApiErrorNotice.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import { DashboardProvider } from './state/DashboardContext.jsx'
import { ThemeProvider } from './state/ThemeContext.jsx'
import { AuthProvider } from './auth/AuthContext.jsx'
import AuthGate from './auth/AuthGate.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <AuthProvider>
        <ThemeProvider>
          <AuthGate>
            {window.location.pathname === '/login' || new URLSearchParams(window.location.search).get('page') === 'login'
              ? <LoginPage />
              : <DashboardProvider><App /><ApiErrorNotice /></DashboardProvider>}
          </AuthGate>
        </ThemeProvider>
      </AuthProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)

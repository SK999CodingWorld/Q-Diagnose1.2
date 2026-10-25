import React from 'react';

export default function AppNavigation({ tabs, activeTab, onChange }) {
  return (
    <nav className="app-navigation-shell border-b px-3 sm:px-7" aria-label="Main navigation">
      <div className="app-navigation mx-auto flex max-w-[1440px] gap-1.5 overflow-x-auto py-2">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onChange(tab.id)}
              aria-current={active ? 'page' : undefined}
              aria-controls={`panel-${tab.id}`}
              title={tab.fullLabel}
              className={`app-nav-button flex items-center gap-2 whitespace-nowrap rounded-xl border px-3.5 py-2.5 text-sm font-medium transition-colors ${active ? 'is-active' : ''}`}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

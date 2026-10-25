import React from 'react';

export default function AppFooter() {
  return (
    <footer className="app-footer mt-auto border-t px-5 py-4">
      <div className="app-footer-inner mx-auto flex max-w-[1440px] flex-col gap-1 text-center text-xs sm:flex-row sm:items-center sm:justify-between sm:text-left">
        <span>Q-Diagnose · Research and decision-support prototype</span>
        <span>Powered by Qiskit Machine Learning and FastAPI</span>
      </div>
    </footer>
  );
}

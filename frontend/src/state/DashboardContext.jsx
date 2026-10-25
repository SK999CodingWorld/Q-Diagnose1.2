import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { datasetsApi } from '../api/datasets';
import { experimentsApi } from '../api/experiments';

const DashboardContext = createContext(null);

export function DashboardProvider({ children }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);

  const refreshDatasets = useCallback(async () => {
    const data = await datasetsApi.list();
    const available = Array.isArray(data.datasets) ? data.datasets : [];
    setDatasets(available);
    setSelectedDataset(current => {
      if (current && available.some(dataset => String(dataset.id) === String(current))) return current;
      const preferred = available.find(dataset => /wisconsin|wdbc/i.test(dataset.name || ''));
      return String(preferred?.id ?? available[0]?.id ?? '');
    });
    return available;
  }, []);

  const refreshExperiments = useCallback(() => setRefreshVersion(version => version + 1), []);

  useEffect(() => {
    refreshDatasets().catch(() => setDatasets([]));
  }, [refreshDatasets]);

  useEffect(() => {
    if (!selectedDataset) {
      setExperiments([]);
      setLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    setLoading(true);
    experimentsApi.list(selectedDataset, { signal: controller.signal })
      .then(data => setExperiments(Array.isArray(data.experiments) ? data.experiments : []))
      .catch(error => {
        if (error.name !== 'AbortError') setExperiments([]);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selectedDataset, refreshVersion]);

  const value = useMemo(() => ({
    datasets,
    selectedDataset,
    setSelectedDataset,
    experiments,
    loading,
    refreshDatasets,
    refreshExperiments,
  }), [datasets, selectedDataset, experiments, loading, refreshDatasets, refreshExperiments]);

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) throw new Error('useDashboard must be used inside DashboardProvider');
  return context;
}

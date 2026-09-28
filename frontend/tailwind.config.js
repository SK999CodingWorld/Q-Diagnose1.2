export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0b1220',
        panel: '#111b2e',
        line: '#263653',
        accent: '#67e8f9',
        violet: '#a78bfa',
        good: '#34d399',
        warn: '#fbbf24',
        bad: '#fb7185',
      },
      boxShadow: {
        glow: '0 0 45px rgba(103, 232, 249, 0.12)',
      },
    },
  },
  plugins: [],
}

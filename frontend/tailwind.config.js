export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#f4f8f3',
        page: '#f4f8f3',
        body: '#20372a',
        panel: '#ffffff',
        line: '#dbe7dc',
        accent: '#16814a',
        violet: '#65a779',
        good: '#16814a',
        warn: '#fbbf24',
        bad: '#fb7185',
      },
      boxShadow: {
        glow: '0 10px 28px rgba(32, 98, 55, 0.10)',
      },
    },
  },
  plugins: [],
}

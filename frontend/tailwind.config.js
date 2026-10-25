/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          navy: '#0A192F',
          dark: '#0B132B',
          card: '#1C2541',
          border: '#3A506B',
          accent: '#64DFDF',
          teal: '#48BFE3',
          purple: '#7209B7',
          pink: '#F72585',
        }
      }
    },
  },
  plugins: [],
}

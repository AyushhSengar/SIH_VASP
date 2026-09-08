import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server port is pinned rather than left to pick a free one: the API's
// CORS_ALLOW_ORIGINS names this exact origin, so a silent shift to 5174 would
// turn every request into an opaque preflight failure. Failing to start is a
// clearer error than that.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
})

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// The built UI is served by the FastAPI service (single container). Relative
// asset paths keep it working behind a reverse-proxy sub-path as well.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2022',
    // No inline scripts or styles: the service's Content-Security-Policy forbids them.
    assetsInlineLimit: 0,
  },
  server: {
    // `npm run dev` proxies API calls to a locally running service.
    proxy: {
      '/health': 'http://127.0.0.1:7860',
      '/ready': 'http://127.0.0.1:7860',
      '/auth': 'http://127.0.0.1:7860',
      '/predict': 'http://127.0.0.1:7860',
    },
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'jsdom',
  },
});

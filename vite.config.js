import { defineConfig } from 'vite';

export default defineConfig({
  // Enable top-level await for transformers.js
  optimizeDeps: {
    exclude: ['@huggingface/transformers']
  },
  build: {
    target: 'esnext'
  },
  server: {
    port: 5173,
    open: true
  }
});

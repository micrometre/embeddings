import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        target: 'esnext',  // Enable top-level await support
    },
    esbuild: {
        target: 'esnext',
    },
    optimizeDeps: {
        esbuildOptions: {
            target: 'esnext',
        },
    },
});
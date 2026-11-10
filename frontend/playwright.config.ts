import { defineConfig, devices } from '@playwright/test';

// Browser tests run the REAL FastAPI service with a SYNTHETIC adapter (a test
// double that "detects" magenta pixels; see tests/e2e_server.py). They check the
// UI and the HTTP contract end to end, not model quality.
const PORT = Number(process.env.E2E_PORT ?? 7861);
const PYTHON = process.env.IMPACTLENS_PYTHON ?? '../.venv/bin/python';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'off',
    acceptDownloads: true,
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: {
    command: `${PYTHON} ../tests/e2e_server.py --port ${PORT} --fixtures e2e/.fixtures`,
    url: `http://127.0.0.1:${PORT}/health`,
    reuseExistingServer: false,
    timeout: 60_000,
    stdout: 'ignore',
  },
});

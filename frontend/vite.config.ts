import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

import { resolveDevServerConfig } from "./devServerConfig";

declare const process: {
  env: Record<string, string | undefined>;
};

const devServer = resolveDevServerConfig(process.env);

export default defineConfig({
  plugins: [react()],
  server: {
    host: devServer.host,
    port: devServer.port,
    proxy: {
      "/api": devServer.backendOrigin,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});

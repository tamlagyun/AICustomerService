import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

import { mergeDevServerEnv, resolveDevServerConfig } from "./devServerConfig";

declare const process: {
  env: Record<string, string | undefined>;
};

export default defineConfig(({ mode }) => {
  const projectEnv = loadEnv(mode, "..", "");
  const devServer = resolveDevServerConfig(mergeDevServerEnv(process.env, projectEnv));

  return {
    plugins: [react()],
    server: {
      host: devServer.host,
      port: devServer.port,
      proxy: devServer.backendProxy,
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
    },
  };
});

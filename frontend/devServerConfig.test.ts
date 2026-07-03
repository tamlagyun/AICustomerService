import { describe, expect, it } from "vitest";

import { mergeDevServerEnv, resolveDevServerConfig } from "./devServerConfig";

describe("resolveDevServerConfig", () => {
  it("uses localhost defaults", () => {
    const config = resolveDevServerConfig({});

    expect(config.host).toBe("127.0.0.1");
    expect(config.port).toBe(5173);
    expect(config.backendOrigin).toBe("http://127.0.0.1:8000");
    expect(config.backendProxy).toEqual({
      "/api": "http://127.0.0.1:8000",
      "/generated": "http://127.0.0.1:8000",
    });
  });

  it("allows host, port, and backend origin to be configured", () => {
    const config = resolveDevServerConfig({
      FRONTEND_HOST: "0.0.0.0",
      FRONTEND_PORT: "6180",
      BACKEND_ORIGIN: "http://192.168.8.151:8000",
    });

    expect(config.host).toBe("0.0.0.0");
    expect(config.port).toBe(6180);
    expect(config.backendOrigin).toBe("http://192.168.8.151:8000");
    expect(config.backendProxy).toEqual({
      "/api": "http://192.168.8.151:8000",
      "/generated": "http://192.168.8.151:8000",
    });
  });

  it("uses project env values when process env is missing", () => {
    const config = resolveDevServerConfig(
      mergeDevServerEnv(
        {},
        {
          BACKEND_ORIGIN: "http://127.0.0.1:8001",
        },
      ),
    );

    expect(config.backendOrigin).toBe("http://127.0.0.1:8001");
    expect(config.backendProxy).toEqual({
      "/api": "http://127.0.0.1:8001",
      "/generated": "http://127.0.0.1:8001",
    });
  });

  it("lets project env override stale process env values", () => {
    const config = resolveDevServerConfig(
      mergeDevServerEnv(
        {
          BACKEND_ORIGIN: "http://127.0.0.1:9000",
        },
        {
          BACKEND_ORIGIN: "http://127.0.0.1:8001",
        },
      ),
    );

    expect(config.backendOrigin).toBe("http://127.0.0.1:8001");
  });
});

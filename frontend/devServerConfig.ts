type EnvLike = Record<string, string | undefined>;

export type DevServerConfig = {
  host: string;
  port: number;
  backendOrigin: string;
};

export function resolveDevServerConfig(env: EnvLike): DevServerConfig {
  return {
    host: env.FRONTEND_HOST || "127.0.0.1",
    port: parsePort(env.FRONTEND_PORT, 5173),
    backendOrigin: env.BACKEND_ORIGIN || "http://127.0.0.1:8000",
  };
}

function parsePort(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  if (!isFinite(parsed) || Math.floor(parsed) !== parsed || parsed <= 0 || parsed > 65535) {
    return fallback;
  }

  return parsed;
}

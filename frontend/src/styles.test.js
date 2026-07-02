import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

describe("chat layout styles", () => {
  const testDir = dirname(fileURLToPath(import.meta.url));
  const styles = readFileSync(resolve(testDir, "styles.css"), "utf-8");

  it("keeps page shell within the viewport and lets only messages scroll", () => {
    expect(styles).toMatch(/box-sizing:\s*border-box/);
    expect(styles).toMatch(/body\s*{[^}]*overflow:\s*hidden/s);
    expect(styles).toMatch(/\.page-shell\s*{[^}]*height:\s*100dvh/s);
    expect(styles).toMatch(/\.chat-panel\s*{[^}]*height:\s*100%/s);
    expect(styles).toMatch(/\.message-list\s*{[^}]*overflow-y:\s*auto/s);
  });
});

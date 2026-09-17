import { defineConfig, globalIgnores } from "eslint/config";
import expoConfig from "eslint-config-expo/flat.js";

export default defineConfig([
  expoConfig,
  globalIgnores([
    ".expo/**",
    ".codex/**",
    ".next/**",
    ".vinext/**",
    ".wrangler/**",
    "android/**",
    "dist/**",
    "ios/**",
    "node_modules/**",
    "sources/**",
    "work/**",
    "outputs/**",
  ]),
]);

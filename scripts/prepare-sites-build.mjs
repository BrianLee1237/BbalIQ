import { copyFile, mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = process.cwd();
const serverDirectory = resolve(root, "dist", "server");
const metadataDirectory = resolve(root, "dist", ".openai");

await mkdir(serverDirectory, { recursive: true });
await mkdir(metadataDirectory, { recursive: true });
await copyFile(
  resolve(root, "hosting", "worker.mjs"),
  resolve(serverDirectory, "index.js"),
);
await copyFile(
  resolve(root, ".openai", "hosting.json"),
  resolve(metadataDirectory, "hosting.json"),
);

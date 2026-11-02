// Runs the app on your own computer: node local.js
// (On Vercel, public/ is served as the site and api/ as functions; this file is not used.)
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { handleApi, json, MODEL } from "./lib/core.js";

const PUBLIC = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");
const PORT = Number(process.env.PORT) || 8000;

if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
  console.error('Missing API key. In PowerShell run:  $env:GEMINI_API_KEY = "your-key"');
  process.exit(1);
}

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon" };

async function serveStatic(urlPath, res) {
  const rel = urlPath === "/" ? "index.html" : urlPath.slice(1);
  const file = path.resolve(PUBLIC, decodeURIComponent(rel));
  if (!file.startsWith(PUBLIC + path.sep)) return json(res, 404, { detail: "Not found." });
  try {
    const data = await fs.readFile(file);
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(data);
  } catch {
    json(res, 404, { detail: "Not found." });
  }
}

http.createServer((req, res) => {
  const { pathname } = new URL(req.url, "http://localhost");
  if (pathname === "/api/debate") return handleApi("debate", req, res);
  if (pathname === "/api/analyze") return handleApi("analyze", req, res);
  if (req.method === "GET") return serveStatic(pathname, res);
  json(res, 405, { detail: "Method not allowed." });
}).listen(PORT, "127.0.0.1", () => {
  console.log(`AI Debate Coach running on http://127.0.0.1:${PORT}  (model: ${MODEL})`);
});

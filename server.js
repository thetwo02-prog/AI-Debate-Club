// AI Debate Coach: Node backend (Gemini).
//
// POST /api/debate   streams the AI opponent's next turn (plain text)
// POST /api/analyze  scores the finished debate and returns coaching feedback
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GoogleGenAI } from "@google/genai";

const STATIC = path.join(path.dirname(fileURLToPath(import.meta.url)), "static");
const PORT = Number(process.env.PORT) || 8000;
// Free-tier friendly default. Override with: $env:GEMINI_MODEL = "other-model-name"
const MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash";

if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
  console.error('Missing API key. In PowerShell run:  $env:GEMINI_API_KEY = "your-key"');
  process.exit(1);
}
const ai = new GoogleGenAI({}); // reads GEMINI_API_KEY (or GOOGLE_API_KEY)

const SIDES = ["for", "against"];
const DIFFICULTIES = ["friendly", "standard", "ruthless"];
const CATEGORIES = ["logic", "evidence", "rebuttals", "confidence", "persuasiveness"];

const DIFFICULTY_STYLE = {
  friendly: "Be encouraging but still honest. Point out one or two weak spots per turn and keep the pressure gentle.",
  standard: "Be a firm, fair opponent. Attack the weakest link in the student's last turn and ask one hard question.",
  ruthless: "Be a relentless, tournament-level opponent. Expose every logical gap, demand evidence, and never concede a point unless it is truly airtight.",
};

function opponentSystem({ topic, student_side, difficulty }) {
  const oppSide = student_side === "for" ? "against" : "for";
  return (
    `You are the opposing debater in a practice debate on the motion: "${topic}".\n` +
    `The student argues ${student_side.toUpperCase()} the motion. You argue ${oppSide.toUpperCase()} it.\n\n` +
    `${DIFFICULTY_STYLE[difficulty]}\n\n` +
    "Rules:\n" +
    "- Stay in role as a debater. Never coach, praise-grade, or break character; feedback comes later.\n" +
    "- Respond directly to what the student actually said: rebut it with a counterargument, " +
    "then end with ONE pointed question they must answer.\n" +
    "- Keep each turn to 80-150 words, in plain paragraphs with no headings or bullet lists.\n" +
    "- Use real-world reasoning and well-known evidence; never invent statistics or quotes. " +
    "If you are unsure of a fact, argue from principle instead.\n" +
    "- If the student's argument is genuinely strong, acknowledge that briefly, then pivot to your best remaining point."
  );
}

const ANALYST_SYSTEM = `You are an experienced debate coach reviewing a practice debate transcript.
Judge ONLY the student's turns (the opponent is an AI sparring partner).

Score each category 0-100 with calibrated, honest standards: 50 is an average beginner, 70 is solid,
85+ is competition-ready. Do not inflate scores.
- logic: validity of reasoning, absence of fallacies, clear claim -> warrant -> impact structure.
- evidence: use of specific facts, examples, and sources versus bare assertion.
- rebuttals: whether the student directly answered the opponent's points and questions, or dodged them.
- confidence: decisive, clear language versus hedging, filler, or giving ground unnecessarily.
- persuasiveness: framing, clarity, rhetorical force, and audience appeal.
overall_score is a weighted judgment, not necessarily the average.

Every piece of feedback must reference what the student actually wrote. Be encouraging in tone but specific
and direct. Exercises must be concrete and directly tied to the weakest categories, e.g. rewrite a specific
weak argument with evidence, or answer an unanswered question from the debate in under 60 seconds.`;

const str = (description) => ({ type: "string", description });
const categorySchema = {
  type: "object",
  properties: {
    score: { type: "integer", description: "0-100" },
    feedback: str("2-3 sentences citing specific moments from the debate"),
  },
  required: ["score", "feedback"],
  additionalProperties: false,
};
const ANALYSIS_SCHEMA = {
  type: "object",
  properties: {
    overall_score: { type: "integer", description: "0-100" },
    verdict: str("One-sentence overall verdict"),
    ...Object.fromEntries(CATEGORIES.map((c) => [c, categorySchema])),
    strengths: { type: "array", items: { type: "string" } },
    weaknesses: { type: "array", items: { type: "string" } },
    best_moment: str("Quote or paraphrase the student's strongest line and why it worked"),
    biggest_missed_opportunity: str("The strongest point the student failed to make or answer"),
    exercises: {
      type: "array",
      description: "3-5 exercises, prioritised by the weakest categories",
      items: {
        type: "object",
        properties: {
          title: { type: "string" },
          targets: { type: "string", enum: CATEGORIES },
          instructions: str("Concrete steps the student can do in 10-15 minutes"),
        },
        required: ["title", "targets", "instructions"],
        additionalProperties: false,
      },
    },
  },
  required: [
    "overall_score", "verdict", ...CATEGORIES, "strengths", "weaknesses",
    "best_moment", "biggest_missed_opportunity", "exercises",
  ],
  additionalProperties: false,
};

// ---------- helpers ----------

class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

function apiErrorText(e) {
  const status = e?.status ?? e?.code;
  const msg = String(e?.message ?? e);
  if (status === 429) return "Rate limited (free tier). Wait a minute and try again.";
  if ([400, 401, 403].includes(status) && /api key/i.test(msg)) {
    return "Your Gemini API key was rejected. Check GEMINI_API_KEY.";
  }
  if (status === 503) return "Google's AI is busy right now. Wait a few seconds and send again.";
  if (status === 404 && /model/i.test(msg)) {
    return `The model "${MODEL}" isn't available. Set a different one with $env:GEMINI_MODEL (see the model list on aistudio.google.com).`;
  }
  return `API error${status ? " " + status : ""}: ${msg}`;
}

// Google answers 503 "high demand" in short spikes, so retry a few times before giving up.
async function retryBusy(fn, tries = 4) {
  for (let i = 1; ; i++) {
    try { return await fn(); }
    catch (e) {
      if ((e?.status ?? e?.code) !== 503 || i >= tries) throw e;
      await new Promise((r) => setTimeout(r, 1500 * i));
    }
  }
}

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const c of req) {
    size += c.length;
    if (size > 1_000_000) throw new HttpError(413, "Request too large.");
    chunks.push(c);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw new HttpError(400, "Invalid JSON."); }
}

function validateTurns(turns, { mustEndWithStudent }) {
  if (!Array.isArray(turns) || turns.length < 1 || turns.length > 40) {
    throw new HttpError(400, "turns must be an array of 1-40 items.");
  }
  for (const t of turns) {
    if (!t || !["student", "opponent"].includes(t.role) ||
        typeof t.text !== "string" || !t.text.length || t.text.length > 4000) {
      throw new HttpError(400, "Each turn needs a role and 1-4000 characters of text.");
    }
  }
  if (turns[0].role !== "student" || (mustEndWithStudent && turns.at(-1).role !== "student")) {
    throw new HttpError(400, "Debate must start and end with a student turn.");
  }
}

function validateSetup(b) {
  if (!b || typeof b.topic !== "string" || b.topic.length < 3 || b.topic.length > 300) {
    throw new HttpError(400, "topic must be 3-300 characters.");
  }
  if (!SIDES.includes(b.student_side)) throw new HttpError(400, "student_side must be 'for' or 'against'.");
  return { ...b, difficulty: b.difficulty ?? "standard" };
}

function json(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

// ---------- routes ----------

async function debate(req, res) {
  const body = await readJson(req);
  const b = validateSetup(body);
  if (!DIFFICULTIES.includes(b.difficulty)) throw new HttpError(400, "Invalid difficulty.");
  validateTurns(b.turns, { mustEndWithStudent: true });

  const contents = b.turns.map((t) => ({
    role: t.role === "student" ? "user" : "model",
    parts: [{ text: t.text }],
  }));

  res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" });
  try {
    let gotText = false;
    const stream = await retryBusy(() => ai.models.generateContentStream({
      model: MODEL,
      contents,
      config: { systemInstruction: opponentSystem(b), maxOutputTokens: 4000 },
    }));
    for await (const chunk of stream) {
      if (chunk.text) { gotText = true; res.write(chunk.text); }
    }
    if (!gotText) res.write("[The opponent declined to respond to this turn. Try rephrasing your argument.]");
  } catch (e) {
    res.write(`\n\n[${apiErrorText(e)}]`);
  }
  res.end();
}

const clampScore = (n) => Math.max(0, Math.min(100, Math.round(Number(n) || 0)));

async function analyze(req, res) {
  const b = validateSetup(await readJson(req));
  validateTurns(b.turns, { mustEndWithStudent: false });

  const transcript = b.turns
    .map((t) => `${t.role === "student" ? "STUDENT" : "OPPONENT"}: ${t.text}`)
    .join("\n\n");
  const prompt =
    `Motion: ${b.topic}\nThe student argued ${b.student_side.toUpperCase()} the motion.\n\n` +
    `<transcript>\n${transcript}\n</transcript>\n\nAnalyze the student's performance.`;

  let result;
  try {
    const resp = await retryBusy(() => ai.models.generateContent({
      model: MODEL,
      contents: prompt,
      config: {
        systemInstruction: ANALYST_SYSTEM,
        maxOutputTokens: 8000,
        responseMimeType: "application/json",
        responseJsonSchema: ANALYSIS_SCHEMA,
      },
    }));
    result = JSON.parse(resp.text);
  } catch (e) {
    if (e instanceof SyntaxError || e instanceof TypeError) {
      throw new HttpError(502, "The analysis could not be generated for this debate. Try again.");
    }
    throw new HttpError(e?.status === 429 ? 429 : 502, apiErrorText(e));
  }

  // The page reads these fields directly, so refuse a malformed result instead of rendering garbage.
  const ok =
    result && CATEGORIES.every((c) => typeof result[c]?.feedback === "string") &&
    Array.isArray(result.strengths) && Array.isArray(result.weaknesses) && Array.isArray(result.exercises);
  if (!ok) throw new HttpError(502, "The analysis came back incomplete. Try again.");

  result.overall_score = clampScore(result.overall_score);
  for (const c of CATEGORIES) result[c].score = clampScore(result[c].score);
  json(res, 200, result);
}

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon" };

async function serveStatic(urlPath, res) {
  const rel = urlPath === "/" ? "index.html" : urlPath.replace(/^\/static\//, "");
  const file = path.resolve(STATIC, rel);
  if (!file.startsWith(STATIC + path.sep)) throw new HttpError(404, "Not found.");
  try {
    const data = await fs.readFile(file);
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream" });
    res.end(data);
  } catch {
    throw new HttpError(404, "Not found.");
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const { pathname } = new URL(req.url, "http://localhost");
    if (req.method === "POST" && pathname === "/api/debate") return await debate(req, res);
    if (req.method === "POST" && pathname === "/api/analyze") return await analyze(req, res);
    if (req.method === "GET") return await serveStatic(pathname, res);
    throw new HttpError(405, "Method not allowed.");
  } catch (e) {
    const status = e instanceof HttpError ? e.status : 500;
    if (!(e instanceof HttpError)) console.error(e);
    if (res.headersSent) return res.end();
    json(res, status, { detail: e instanceof HttpError ? e.message : "Server error." });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`AI Debate Coach running on http://127.0.0.1:${PORT}  (model: ${MODEL})`);
});

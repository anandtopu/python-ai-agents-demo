import React, { useEffect, useMemo, useState } from "react";

type RunResponse = {
  demo: string;
  run_id?: string;
  events: Array<Record<string, unknown>>;
  outputs: Record<string, unknown>;
  evaluation?: unknown;
};

type RunsResponse = {
  runs: Array<{ run_id: string; path: string; mtime: number }>;
};

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [demos, setDemos] = useState<string[]>([]);
  const [demo, setDemo] = useState<string>("context");
  const [evaluate, setEvaluate] = useState<boolean>(true);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [runs, setRuns] = useState<Array<{ run_id: string; path: string; mtime: number }>>([]);

  const [agentFilter, setAgentFilter] = useState<string>("all");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [search, setSearch] = useState<string>("");

  const sortedDemos = useMemo(() => [...demos].sort(), [demos]);

  useEffect(() => {
    void (async () => {
      try {
        const rr = await fetch(`${API_BASE}/api/runs`);
        if (rr.ok) {
          const rdata = (await rr.json()) as RunsResponse;
          setRuns(rdata.runs ?? []);
        }

        const r = await fetch(`${API_BASE}/api/demos`);
        if (!r.ok) {
          throw new Error(`HTTP ${r.status}`);
        }
        const data = (await r.json()) as { demos: string[] };
        setDemos(data.demos);
        if (data.demos.includes("context")) {
          setDemo("context");
        } else if (data.demos.length > 0) {
          setDemo(data.demos[0]);
        }
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  async function onLoadRun(runId: string) {
    setRunning(true);
    setError("");
    try {
      const r = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(runId)}`);
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      const data = (await r.json()) as { run_id: string; events: Array<Record<string, unknown>> };
      setResult({ demo: "(loaded)", run_id: data.run_id, events: data.events, outputs: {}, evaluation: null });
      setAgentFilter("all");
      setTypeFilter("all");
      setSearch("");
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  async function onRun() {
    setRunning(true);
    setError("");
    setResult(null);

    try {
      const r = await fetch(`${API_BASE}/api/run`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ demo, evaluate }),
      });
      const data = (await r.json()) as RunResponse | { detail: string };
      if (!r.ok) {
        const msg = (data as { detail: string }).detail || `HTTP ${r.status}`;
        throw new Error(msg);
      }
      setResult(data as RunResponse);
      setAgentFilter("all");
      setTypeFilter("all");
      setSearch("");
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui", padding: 16, maxWidth: 1100, margin: "0 auto" }}>
      <h1 style={{ marginTop: 0 }}>Multi-Agent Demo UI</h1>

      <div style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <strong>Runs</strong>
        <select
          value=""
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
            const v = e.target.value;
            if (v) {
              void onLoadRun(v);
            }
          }}
          style={{ padding: 6, minWidth: 240 }}
        >
          <option value="">Load past run…</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id}
            </option>
          ))}
        </select>
        {result?.run_id ? (
          <span>
            Current run: <code>{result.run_id}</code>
          </span>
        ) : null}
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <label>
          Demo
          <select
            value={demo}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setDemo(e.target.value)}
            style={{ marginLeft: 8, padding: 6 }}
          >
            {sortedDemos.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={evaluate}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEvaluate(e.target.checked)}
          />
          Evaluate
        </label>

        <button onClick={onRun} disabled={running} style={{ padding: "8px 12px" }}>
          {running ? "Running..." : "Run"}
        </button>

        {result ? (
          <>
            <label>
              Agent
              <select
                value={agentFilter}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setAgentFilter(e.target.value)}
                style={{ marginLeft: 8, padding: 6 }}
              >
                <option value="all">all</option>
                {uniqueAgents(result.events).map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Type
              <select
                value={typeFilter}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setTypeFilter(e.target.value)}
                style={{ marginLeft: 8, padding: 6 }}
              >
                <option value="all">all</option>
                {uniqueTypes(result.events).map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Search
              <input
                value={search}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
                placeholder="tool name, agent, text…"
                style={{ marginLeft: 8, padding: 6, width: 220 }}
              />
            </label>
          </>
        ) : null}
      </div>

      {error ? (
        <div style={{ marginTop: 12, padding: 12, background: "#fee2e2", border: "1px solid #fecaca" }}>
          <strong>Error:</strong> {error}
        </div>
      ) : null}

      {result ? (
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <section style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Timeline</h2>
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {filterEvents(result.events, agentFilter, typeFilter, search).map((ev: Record<string, unknown>, idx: number) => (
                <li key={idx}>
                  <EventRow ev={ev} />
                </li>
              ))}
            </ol>
          </section>

          <section style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Evaluation</h2>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              {JSON.stringify(result.evaluation ?? null, null, 2)}
            </pre>
          </section>

          <section style={{ gridColumn: "1 / -1", border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Outputs</h2>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(result.outputs, null, 2)}</pre>
          </section>
        </div>
      ) : null}

      <p style={{ marginTop: 16, color: "#6b7280" }}>
        Backend expected at <code>{API_BASE}</code>. Start it with <code>uvicorn backend.main:app --reload</code>.
      </p>
    </div>
  );
}

function EventRow({ ev }: { ev: Record<string, unknown> }) {
  const type = String(ev.type ?? "event");
  const agent = ev.agent ? String(ev.agent) : "";

  const header = (
    <span>
      {agent ? <Badge text={agent} background={agentColor(agent)} /> : null}{" "}
      <Badge text={type} background={typeColor(type)} />
    </span>
  );

  if (type === "user_message") {
    return (
      <span>
        {header} {String(ev.content ?? "")}
      </span>
    );
  }

  if (type === "assistant_message") {
    const content = String(ev.content ?? "");
    const toolCalls = ev.tool_calls ? " (tool calls)" : "";
    return (
      <span>
        {header}
        {" "}
        <span>
          assistant{toolCalls}: <InlineExpandableText text={content} />
        </span>
      </span>
    );
  }

  if (type === "tool_call") {
    const args = ev.args as unknown;
    const name = String(ev.name ?? "");
    const argsText = safeJson(args);
    return (
      <details>
        <summary>
          {header} tool_call: <code>{name}</code>
        </summary>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <button onClick={() => void copyText(argsText)} style={{ padding: "4px 8px" }}>
            Copy args
          </button>
        </div>
        <pre style={{ margin: "8px 0 0", whiteSpace: "pre-wrap" }}>{argsText}</pre>
      </details>
    );
  }

  if (type === "tool_result") {
    const result = ev.result as unknown;
    const name = String(ev.name ?? "");
    const resultText = safeJson(result);
    return (
      <details>
        <summary>
          {header} tool_result: <code>{name}</code>
        </summary>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <button onClick={() => void copyText(resultText)} style={{ padding: "4px 8px" }}>
            Copy result
          </button>
        </div>
        <pre style={{ margin: "8px 0 0", whiteSpace: "pre-wrap" }}>{resultText}</pre>
      </details>
    );
  }

  if (type === "llm_request") {
    return (
      <span>
        {header} llm_request
      </span>
    );
  }

  return (
    <span>
      {header} <InlineExpandableText text={safeJson(ev)} limit={160} />
    </span>
  );
}

function InlineExpandableText({ text, limit = 240 }: { text: string; limit?: number }) {
  if (text.length <= limit) {
    return <span>{text}</span>;
  }

  const head = text.slice(0, limit);
  return (
    <details style={{ display: "inline" }}>
      <summary style={{ display: "inline", cursor: "pointer" }}>{head}…</summary>
      <span> {text}</span>
    </details>
  );
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function uniqueAgents(events: Array<Record<string, unknown>>): string[] {
  const set = new Set<string>();
  for (const ev of events) {
    if (ev.agent) {
      set.add(String(ev.agent));
    }
  }
  return [...set].sort();
}

function uniqueTypes(events: Array<Record<string, unknown>>): string[] {
  const set = new Set<string>();
  for (const ev of events) {
    if (ev.type) {
      set.add(String(ev.type));
    }
  }
  return [...set].sort();
}

function filterEvents(
  events: Array<Record<string, unknown>>,
  agentFilter: string,
  typeFilter: string,
  search: string
): Array<Record<string, unknown>> {
  const q = search.trim().toLowerCase();
  return events.filter((ev) => {
    const agentOk = agentFilter === "all" || String(ev.agent ?? "") === agentFilter;
    const typeOk = typeFilter === "all" || String(ev.type ?? "") === typeFilter;
    if (!agentOk || !typeOk) {
      return false;
    }
    if (!q) {
      return true;
    }
    const hay = safeJson(ev).toLowerCase();
    return hay.includes(q);
  });
}

async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // ignore
  }
}

function Badge({ text, background }: { text: string; background: string }) {
  return (
    <span
      style={{
        background,
        color: "#111827",
        border: "1px solid rgba(17,24,39,0.15)",
        padding: "1px 6px",
        borderRadius: 999,
        fontSize: 12,
        whiteSpace: "nowrap",
      }}
    >
      {text}
    </span>
  );
}

function typeColor(type: string): string {
  switch (type) {
    case "user_message":
      return "#dbeafe";
    case "assistant_message":
      return "#dcfce7";
    case "tool_call":
      return "#ffedd5";
    case "tool_result":
      return "#fef9c3";
    case "llm_request":
      return "#e5e7eb";
    default:
      return "#f3f4f6";
  }
}

function agentColor(agent: string): string {
  const colors = ["#e0e7ff", "#fae8ff", "#cffafe", "#ffe4e6", "#ecfccb", "#fce7f3"];
  let h = 0;
  for (let i = 0; i < agent.length; i++) {
    h = (h * 31 + agent.charCodeAt(i)) >>> 0;
  }
  return colors[h % colors.length];
}

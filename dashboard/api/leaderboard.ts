import type { VercelRequest, VercelResponse } from "@vercel/node";

// Server-only proxy: PULSO_API_KEY never reaches the browser. Combines /v1/me and both
// leaderboard windows into one response so the client only makes a single request.
export default async function handler(req: VercelRequest, res: VercelResponse) {
  const baseUrl = (process.env.PULSO_API_URL ?? "https://pulso-transmi.72-60-245-2.sslip.io").replace(/\/$/, "");
  const apiKey = process.env.PULSO_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "PULSO_API_KEY is not configured on the server." });
    return;
  }

  const headers = { Authorization: `Bearer ${apiKey}` };

  try {
    const [meResponse, cumulativeResponse, rolling24hResponse] = await Promise.all([
      fetch(`${baseUrl}/v1/me`, { headers }),
      fetch(`${baseUrl}/v1/leaderboard?window=cumulative`, { headers }),
      fetch(`${baseUrl}/v1/leaderboard?window=rolling_24h`, { headers }),
    ]);

    if (!meResponse.ok || !cumulativeResponse.ok || !rolling24hResponse.ok) {
      const failed = [meResponse, cumulativeResponse, rolling24hResponse].find((response) => !response.ok);
      const detail = failed ? await failed.text() : "unknown error";
      res.status(502).json({ error: `Pulso API request failed: ${detail}` });
      return;
    }

    const [me, cumulative, rolling24h] = await Promise.all([
      meResponse.json(),
      cumulativeResponse.json(),
      rolling24hResponse.json(),
    ]);

    res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=120");
    res.status(200).json({ me, cumulative: cumulative.data ?? [], rolling_24h: rolling24h.data ?? [] });
  } catch (error) {
    res.status(502).json({ error: error instanceof Error ? error.message : "Unknown error contacting the Pulso API." });
  }
}

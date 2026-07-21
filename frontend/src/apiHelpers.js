// Shared client-side helpers for the async API flows (analyze + walkthrough).

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A cold first request typically fails as an API Gateway 504 (integration
// timeout past the 29s cap), a 503 from a warming instance, or a dropped
// connection (ERR_NETWORK / ECONNABORTED). Those warrant one automatic retry;
// a plain application error (e.g. a 4xx/5xx bug) does not, so it is NOT retried.
export function isColdStartError(err) {
  const status = err?.response?.status;
  if (status === 503 || status === 504) return true;
  return err?.code === "ERR_NETWORK" || err?.code === "ECONNABORTED";
}

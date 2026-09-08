/**
 * The only module that talks to the API.
 *
 * It deliberately does NOT reshape, default, or infer anything. The backend's
 * `InvestigationBrief` is already the projection meant for a reader; inventing
 * a client-side default for a missing field would put a number on screen that
 * no investigation produced. A field that is null renders as "N/A", and that
 * decision lives in the components, not here.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

/** An error carrying the API's own error code and message. */
export class ApiError extends Error {
  constructor(code, detail, status) {
    super(detail)
    this.name = 'ApiError'
    this.code = code
    this.detail = detail
    this.status = status
  }
}

/**
 * Runs one investigation and returns the brief.
 *
 * There is no client-side timeout. A live acquisition expands hop by hop and
 * legitimately takes minutes; aborting it at some arbitrary deadline would
 * report a failure for a run that was working, and the user would have no way
 * to tell that apart from a real one. The UI says the wait is expected instead.
 */
export async function fetchBrief({ wallet, chain, preferCached }, signal) {
  let response
  try {
    response = await fetch(`${BASE_URL}/analysis/brief`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        wallet: wallet.trim(),
        chain,
        prefer_cached: preferCached,
      }),
      signal,
    })
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause
    // fetch rejects only when the request never got an HTTP reply at all: the
    // API is down, the port is wrong, or CORS refused the origin. Naming those
    // three is more useful than "failed to fetch", which sends people looking
    // at the wallet address.
    throw new ApiError(
      'API_UNREACHABLE',
      `Could not reach the API at ${BASE_URL}. Check that it is running ` +
        `(python -m uvicorn app.api.main:app --reload) and that this origin is ` +
        `listed in CORS_ALLOW_ORIGINS.`,
      0,
    )
  }

  let body = null
  try {
    body = await response.json()
  } catch {
    body = null
  }

  if (!response.ok) {
    // FastAPI's own request-validation errors use `detail`; ours use
    // {error, detail}. Both are surfaced verbatim — the backend writes these
    // messages to be read, and paraphrasing them here would lose the reason.
    const code = body?.error || `HTTP_${response.status}`
    let detail = body?.detail
    if (Array.isArray(detail)) {
      detail = detail.map((item) => item?.msg || JSON.stringify(item)).join('; ')
    }
    throw new ApiError(code, detail || `The API returned ${response.status}.`, response.status)
  }

  return body
}

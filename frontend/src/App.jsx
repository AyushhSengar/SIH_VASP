import { useEffect, useRef, useState } from 'react'
import { ApiError, fetchBrief } from './api.js'
import Brief from './Brief.jsx'
import {
    BarChartIcon,
    ChainIcon,
    CheckIcon,
    CopyIcon,
    EthIcon,
    FileSearchIcon,
    InfoIcon,
    LockIcon,
    MoonIcon,
    SearchIcon,
    ShareNodesIcon,
    ShieldIcon,
    SpinnerIcon,
    SunIcon,
    WalletIcon,
} from './Icons.jsx'

// The chains the API can resolve. Only ethereum has been exercised against
// live provider data in this build, which the option text says rather than
// leaving the reader to find out from an error.
const CHAINS = [
  'ethereum',
//   'polygon',
//   'bsc',
//   'bitcoin'
//   'arbitrum',
//   'optimism',
//   'base',
//   'avalanche',
]

const FEATURES = [
  {
    icon: FileSearchIcon,
    tone: 'blue',
    title: 'Detect Risk',
    body: 'Identify suspicious activity',
  },
  {
    icon: ShareNodesIcon,
    tone: 'green',
    title: 'Trace Connections',
    body: 'Uncover hidden links',
  },
  {
    icon: BarChartIcon,
    tone: 'orange',
    title: 'On-Chain Insights',
    body: 'Data-driven analysis',
  },
  {
    icon: LockIcon,
    tone: 'red',
    title: 'Make Safer Decisions',
    body: 'Investigate before you transact',
  },
]

function useTheme() {
  const [theme, setTheme] = useState(() => {
    const saved = typeof window !== 'undefined' ? window.localStorage.getItem('theme') : null
    if (saved === 'light' || saved === 'dark') return saved
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark'
    }
    return 'light'
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('theme', theme)
  }, [theme])

  return [theme, setTheme]
}

export default function App() {
  const [theme, setTheme] = useTheme()
  const [wallet, setWallet] = useState('')
  const [chain, setChain] = useState('ethereum')
  const [preferCached, setPreferCached] = useState(true)
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [brief, setBrief] = useState(null)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const abortRef = useRef(null)

  async function submit(event) {
    event.preventDefault()
    if (!wallet.trim() || status === 'loading') return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('loading')
    setError(null)
    // The previous result is cleared before the new one is requested. Leaving
    // it on screen under a spinner invites reading a stale verdict as the
    // answer to the address just typed.
    setBrief(null)

    try {
      const result = await fetchBrief({ wallet, chain, preferCached }, controller.signal)
      setBrief(result)
      setStatus('done')
    } catch (cause) {
      if (cause?.name === 'AbortError') return
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError('UNEXPECTED', cause?.message || 'Something went wrong.', 0),
      )
      setStatus('error')
    }
  }

  function cancel() {
    abortRef.current?.abort()
    setStatus('idle')
  }

  async function copyWallet() {
    if (!wallet.trim()) return
    try {
      await navigator.clipboard.writeText(wallet.trim())
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      setCopied(false)
    }
  }

  return (
    <div className="page">
      <nav className="navbar">
        <div className="brand">
          <span className="brand-mark">
            <ShieldIcon />
          </span>
          <span className="brand-text">
            CRYPTO <span className="accent-text">SCOUT</span>
          </span>
        </div>
        {/* <div className="nav-links">
          <span className="nav-link nav-link-active">Investigate</span>
          <span className="nav-link nav-link-muted">Analytics</span>
          <span className="nav-link nav-link-muted">About</span>
        </div> */}
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <MoonIcon width={16} height={16} /> : <SunIcon width={16} height={16} />}
        </button>
      </nav>

      <header className="hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          On-chain intelligence
        </span>
        <h1>
          Crypto Wallet <span className="accent-text">Investigation</span>
        </h1>
        <p className="hero-sub">Analyze wallet behavior. Detect risks. Make smarter decisions.</p>
      </header>

      <form className="card query" onSubmit={submit}>
        <div className="row">
          <label className="grow field-block">
            <span className="field-label">
              <WalletIcon width={16} height={16} />
              Wallet Address
            </span>
            <div className="input-wrap">
              <input
                className="mono"
                type="text"
                value={wallet}
                onChange={(event) => setWallet(event.target.value)}
                placeholder="0x followed by 40 hexadecimal characters"
                spellCheck="false"
                autoComplete="off"
                autoCapitalize="off"
                aria-label="Wallet address"
              />
              <button
                type="button"
                className="input-icon-btn"
                onClick={copyWallet}
                aria-label="Copy wallet address"
                disabled={!wallet.trim()}
              >
                {copied ? <CheckIcon width={16} height={16} /> : <CopyIcon width={16} height={16} />}
              </button>
            </div>
          </label>

          <label className="field-block chain-block">
            <span className="field-label">
              <ChainIcon width={16} height={16} />
              Chain
            </span>
            <div className="select-wrap">
              <EthIcon className="select-icon" />
              <select value={chain} onChange={(event) => setChain(event.target.value)}>
                {CHAINS.map((name) => (
                  <option key={name} value={name}>
                    {name.charAt(0).toUpperCase() + name.slice(1)}
                    {name === 'ethereum' ? '' : ' (not live-validated)'}
                  </option>
                ))}
              </select>
            </div>
          </label>
        </div>

        <div className="row bottom">
          <div className="check-block">
            <label className="check">
              <input
                type="checkbox"
                checked={preferCached}
                onChange={(event) => setPreferCached(event.target.checked)}
              />
              <span>
                Reuse real data already cached for this wallet
                <InfoIcon
                  className="check-info"
                  aria-label="Uncheck to fetch live, which expands hop by hop and can take several minutes"
                />
              </span>
            </label>
            <p className="check-sub">
              Uncheck to fetch live, which expands hop by hop and can take several minutes.
            </p>
          </div>

          <div className="actions">
            {status === 'loading' && (
              <button type="button" className="btn-ghost" onClick={cancel}>
                Cancel
              </button>
            )}
            <button type="submit" className="btn-primary" disabled={!wallet.trim() || status === 'loading'}>
              <SearchIcon width={17} height={17} />
              {status === 'loading' ? 'Investigating' : 'Investigate'}
            </button>
          </div>
        </div>
      </form>

      {status === 'loading' && (
        <div className="card state loading">
          <div className="spinner-ring">
            <SpinnerIcon width={26} height={26} />
          </div>
          <div className="state-body">
            <strong>Running the investigation&hellip;</strong>
            <p>
              {preferCached
                ? 'Loading the cached real graph and running every analysis stage.'
                : 'Fetching live blockchain data hop by hop. This can take several minutes; no demo or synthetic data is substituted if it fails.'}
            </p>
          </div>
          <div className="state-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </div>
      )}

      {status === 'error' && error && (
        <div className="card state error">
          <div className="state-body">
            <strong>
              {error.code === 'INVESTIGATION_STOPPED'
                ? 'The investigation stopped'
                : 'That request could not be answered'}
            </strong>
            <p>{error.detail}</p>
            <p className="mono muted">
              {error.code}
              {error.status ? ` · HTTP ${error.status}` : ''}
            </p>
          </div>
        </div>
      )}

      {status === 'done' && brief && <Brief brief={brief} />}

      {/* {status === 'idle' && (
        <div className="card state hint">
          <div className="state-body">
            <strong>Enter a wallet address to begin.</strong>
            <p>
              A result is labelled with the data it came from: <span className="mono">REAL</span>{' '}
              for a live fetch, <span className="mono">CACHED REAL DATA</span> for real records
              already on disk. Nothing is ever substituted for missing data &mdash; a run that
              cannot get real data fails instead.
            </p>
          </div>
        </div>
      )} */}

      {/* <section className="features">
        {FEATURES.map(({ icon: Icon, tone, title, body }) => (
          <div className="feature" key={title}>
            <span className={`feature-icon feature-icon-${tone}`}>
              <Icon width={22} height={22} />
            </span>
            <div>
              <div className="feature-title">{title}</div>
              <div className="feature-body">{body}</div>
            </div>
          </div>
        ))}
      </section> */}

      <footer className="page-foot">
        <span className="foot-rule" />
        <span>
          Built From Scratch By <strong className="accent-text">CRYPTO SCOUT</strong>
        </span>
        <span className="foot-rule" />
      </footer>
    </div>
  )
}
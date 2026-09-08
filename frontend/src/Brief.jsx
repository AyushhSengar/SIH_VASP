import { useState } from 'react'
import { CheckIcon, CopyIcon } from './Icons.jsx'

/** Absent means absent. Never a zero, never an empty string, never a guess. */
const NA = 'N/A'

function value(raw) {
  if (raw === null || raw === undefined || raw === '') return NA
  return String(raw)
}

/**
 * The three-state answer, kept three-state.
 *
 * `matched` is true / false / null and the null case is INCONCLUSIVE — a search
 * that was cut short before it could answer. Collapsing that to "NO" would turn
 * "we could not finish looking" into "there was nothing there", which is the one
 * mistake this whole view exists to avoid.
 */
function verdict(match) {
  if (match.matched === true) return { word: 'YES', tone: 'match' }
  if (match.matched === false) return { word: 'NO', tone: 'clear' }
  if (match.status === 'NOT_RUN') return { word: 'NOT RUN', tone: 'unknown' }
  return { word: 'INCONCLUSIVE', tone: 'unknown' }
}

/**
 * Monospaced address or hash with a copy button.
 *
 * Deliberately not a link to a block explorer: this is an investigation tool,
 * and following a link would disclose the address under investigation to a third
 * party. Copying keeps it local, and is what an investigator does with it next
 * anyway.
 */
function Hash({ text, title }) {
  const [copied, setCopied] = useState(false)

  if (!text) return <span className="mono muted">{NA}</span>

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      // Clipboard access can be refused (insecure origin, denied permission).
      // The text is on screen and selectable, so nothing is lost; claiming a
      // successful copy that did not happen would be worse.
      setCopied(false)
    }
  }

  return (
    <span className="hash" title={title}>
      <span className="mono">{text}</span>
      <button type="button" className="copy" onClick={copy} aria-label={`Copy ${text}`}>
        {copied ? <CheckIcon width={13} height={13} /> : <CopyIcon width={13} height={13} />}
      </button>
    </span>
  )
}

function Field({ label, children }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      <span className="value">{children}</span>
    </div>
  )
}

export default function Brief({ brief }) {
  const match = brief.vasp_match
  const { word, tone } = verdict(match)
  const path = match.evidence_path || []

  return (
    <article className="card result">
      <header className="result-head">
        <div>
          <span className="label">Wallet</span>
          <div className="wallet">
            <Hash text={brief.wallet} title="The investigated wallet" />
          </div>
        </div>
        <div className="chips">
          <span className="chip">{value(brief.chain)}</span>
          <span className="chip" title="Where this run's data came from">
            {value(brief.data_mode)}
          </span>
          <span className="chip muted">{value(brief.duration_seconds)}s</span>
        </div>
      </header>

      <section className={`verdict verdict-${tone}`}>
        <div className="verdict-word">{word}</div>
        <div className="verdict-body">
          <div className="verdict-title">
            VASP match <span className="mono muted">({value(match.status)})</span>
          </div>
          {match.wallet_is_known_vasp && (
            <p className="identity">
              The wallet itself is a dataset address: <strong>{match.wallet_is_known_vasp}</strong>
            </p>
          )}
          {(match.vasp_name || match.direction || match.hop_distance !== null) && (
            <div className="fields">
              <Field label="Entity">{value(match.vasp_name)}</Field>
              <Field label="Direction">
                <span className="mono">{value(match.direction)}</span>
              </Field>
              <Field label="Hops">{value(match.hop_distance)}</Field>
              <Field label="Source">
                <span className="mono">{value(match.source_type)}</span>
              </Field>
            </div>
          )}
          {match.note && <p className="note">{match.note}</p>}
        </div>
      </section>

      {path.length > 0 && (
        <section className="block">
          <h2>Evidence path</h2>
          <ol className="path">
            {path.map((address, index) => (
              <li key={`${address}-${index}`}>
                <Hash text={address} />
                {index < path.length - 1 && (
                  <div className="edge">
                    <span className="label">tx</span>
                    <Hash
                      text={(match.evidence_tx_hashes || [])[index]}
                      title="The transfer from this address to the next"
                    />
                  </div>
                )}
              </li>
            ))}
          </ol>
        </section>
      )}

      {brief.risk_summary?.length > 0 && (
        <section className="block">
          <h2>Risk and behaviour</h2>
          {brief.risk_summary.map((line, index) => (
            <p key={index} className="summary-line">
              {line}
            </p>
          ))}
        </section>
      )}

      <section className="block">
        <h2>
          Machine learning <span className="mono muted">({value(brief.ml.approach)})</span>
        </h2>
        <p className="summary-line">{brief.ml.verdict}</p>
        {brief.ml.model_name && (
          <p className="metrics mono">
            {brief.ml.model_name} {value(brief.ml.model_version)} &middot; held-out F1{' '}
            {value(brief.ml.held_out_f1)} &middot; accuracy {value(brief.ml.held_out_accuracy)}
          </p>
        )}
        <p className="disclaimer">{brief.ml.disclaimer}</p>
      </section>

      {brief.warnings?.length > 0 && (
        <section className="block">
          <h2>Warnings</h2>
          {brief.warnings.map((warning, index) => (
            <p key={index} className="summary-line warning">
              {warning}
            </p>
          ))}
        </section>
      )}

      <footer className="result-foot mono muted">{value(brief.investigation_id)}</footer>
    </article>
  )
}
/**
 * Inline SVG icons.
 *
 * Kept as plain inline SVG rather than an icon-library dependency: this
 * project has no package for one yet, and eight icons don't justify adding
 * one. Every icon is stroke-based, 1.8px, currentColor, so it always matches
 * the text color around it without a separate color prop.
 */

const base = {
  width: 20,
  height: 20,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

export function ShieldIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  )
}

export function SunIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4" />
    </svg>
  )
}

export function MoonIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" />
    </svg>
  )
}

export function WalletIcon(props) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="6" width="18" height="13" rx="2" />
      <path d="M3 10h18" />
      <path d="M16 14h2" />
    </svg>
  )
}

export function ChainIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="6" cy="6" r="2.4" />
      <circle cx="18" cy="6" r="2.4" />
      <circle cx="12" cy="18" r="2.4" />
      <path d="M8 7.2L10.5 16M16 7.2L13.5 16" />
    </svg>
  )
}

export function EthIcon(props) {
  return (
    <svg {...base} viewBox="0 0 16 16" width={16} height={16} {...props}>
      <path d="M8 1l4.8 7.6L8 11.2 3.2 8.6 8 1z" fill="currentColor" stroke="none" opacity="0.55" />
      <path d="M8 12.2l4.8-3.6L8 15l-4.8-6.4 4.8 3.6z" fill="currentColor" stroke="none" />
    </svg>
  )
}

export function CopyIcon(props) {
  return (
    <svg {...base} {...props}>
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
    </svg>
  )
}

export function CheckIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M4 12l5 5L20 6" />
    </svg>
  )
}

export function InfoIcon(props) {
  return (
    <svg {...base} width={16} height={16} {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5.5M12 8v.01" />
    </svg>
  )
}

export function SearchIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.3-4.3" />
    </svg>
  )
}

export function SpinnerIcon(props) {
  return (
    <svg {...base} className="spin" {...props}>
      <path d="M12 3a9 9 0 1 0 9 9" />
    </svg>
  )
}

export function FileSearchIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
      <circle cx="10.5" cy="14" r="2.3" />
      <path d="M12.3 15.8L14 17.5" />
    </svg>
  )
}

export function ShareNodesIcon(props) {
  return (
    <svg {...base} {...props}>
      <circle cx="6" cy="6" r="2.3" />
      <circle cx="6" cy="18" r="2.3" />
      <circle cx="18" cy="12" r="2.3" />
      <path d="M8 6.9L16 11M8 17.1L16 13" />
    </svg>
  )
}

export function BarChartIcon(props) {
  return (
    <svg {...base} {...props}>
      <path d="M4 20V10M11 20V4M18 20v-7" />
    </svg>
  )
}

export function LockIcon(props) {
  return (
    <svg {...base} {...props}>
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </svg>
  )
}
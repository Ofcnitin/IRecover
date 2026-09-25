import { SunIcon, MoonIcon, LockIcon, UnlockIcon } from './icons';
import type { PrivacyMode } from '../types/photoFinish';

interface HeaderProps {
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  privacyMode: PrivacyMode;
}

const NAV_ITEMS = [
  { id: 'convert', label: 'Convert' },
  { id: 'examples', label: 'Examples' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'about', label: 'About' },
  { id: 'faq', label: 'FAQ' },
];

export default function Header({ theme, onToggleTheme, privacyMode }: HeaderProps) {
  const isPrivate = privacyMode === 'local';

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <header className="site-header">
      <div className="site-header__brand">
        <span className="brand-mark" aria-hidden="true">
          <span className="brand-mark__dot brand-mark__dot--a" />
          <span className="brand-mark__dot brand-mark__dot--b" />
        </span>
        <div>
          <div className="brand-name">IRecover</div>
          <div className="brand-tagline">From Infrared to a Clearer World</div>
        </div>
      </div>

      <nav className="site-header__nav" aria-label="Primary">
        {NAV_ITEMS.map((item, i) => (
          <button
            key={item.id}
            type="button"
            className={`nav-link ${i === 0 ? 'nav-link--active' : ''}`}
            onClick={() => scrollTo(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="site-header__actions">
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          aria-pressed={theme === 'dark'}
        >
          <SunIcon size={16} className={theme === 'light' ? 'theme-toggle__icon theme-toggle__icon--active' : 'theme-toggle__icon'} />
          <MoonIcon size={16} className={theme === 'dark' ? 'theme-toggle__icon theme-toggle__icon--active' : 'theme-toggle__icon'} />
        </button>

        <div
          className={`privacy-pill ${isPrivate ? 'privacy-pill--local' : 'privacy-pill--cloud'}`}
          role="status"
          title={
            isPrivate
              ? 'Processed locally in your browser.'
              : 'Photo Finish sends the converted image to the configured Cloudinary service for final processing.'
          }
        >
          {isPrivate ? <LockIcon size={16} /> : <UnlockIcon size={16} />}
          <div className="privacy-pill__text">
            <span className="privacy-pill__title">{isPrivate ? '100% Private' : 'Photo Finish Active'}</span>
            <span className="privacy-pill__subtitle">
              {isPrivate ? 'Processed locally in your browser' : 'Sent to Cloudinary for finishing'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

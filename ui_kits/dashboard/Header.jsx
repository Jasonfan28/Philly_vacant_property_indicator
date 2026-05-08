// Header.jsx — Philly Stat 360 top navigation bar

const Header = ({ activePage, onNavigate }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'departments', label: 'Departments' },
    { id: 'data', label: 'Open Data' },
    { id: 'about', label: 'About' },
  ];

  return (
    <header style={headerStyles.bar}>
      <div style={headerStyles.inner}>
        <button onClick={() => onNavigate('dashboard')} style={headerStyles.logoBtn}>
          <img src="../../assets/logo-blue-text.png" height="36" alt="City of Philadelphia" />
          <span style={headerStyles.appName}>Philly Stat 360</span>
        </button>
        <nav>
          <ul style={headerStyles.navList}>
            {navItems.map(item => (
              <li key={item.id}>
                <button
                  onClick={() => onNavigate(item.id)}
                  style={{
                    ...headerStyles.navLink,
                    color: activePage === item.id ? '#2176d2' : '#444',
                    borderBottom: activePage === item.id ? '3px solid #f3c613' : '3px solid transparent',
                  }}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
};

const headerStyles = {
  bar: {
    background: '#fff',
    borderBottom: '1px solid #cfcfcf',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  inner: {
    maxWidth: 1200,
    margin: '0 auto',
    padding: '0 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 64,
  },
  logoBtn: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: 0,
  },
  appName: {
    fontFamily: "'Montserrat', sans-serif",
    fontWeight: 700,
    fontSize: 16,
    color: '#0f4d90',
    letterSpacing: '-0.2px',
  },
  navList: {
    display: 'flex',
    listStyle: 'none',
    margin: 0,
    padding: 0,
    gap: 0,
  },
  navLink: {
    display: 'flex',
    alignItems: 'center',
    height: 64,
    padding: '0 16px',
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 13,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    background: 'none',
    border: 'none',
    borderBottom: '3px solid transparent',
    cursor: 'pointer',
    transition: 'color 0.15s, border-color 0.15s',
    whiteSpace: 'nowrap',
  },
};

Object.assign(window, { Header });

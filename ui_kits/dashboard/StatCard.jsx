// StatCard.jsx — KPI stat card component

const StatCard = ({ label, value, unit, trend, trendDir, target, onClick }) => {
  const trendColor = trendDir === 'up' ? '#3a833c' : trendDir === 'down' ? '#cc3000' : '#a1a1a1';
  const trendArrow = trendDir === 'up' ? '▲' : trendDir === 'down' ? '▼' : '–';

  return (
    <div
      onClick={onClick}
      style={{
        ...statCardStyles.card,
        cursor: onClick ? 'pointer' : 'default',
      }}
      onMouseEnter={e => { if (onClick) e.currentTarget.style.borderColor = '#2176d2'; }}
      onMouseLeave={e => { if (onClick) e.currentTarget.style.borderColor = '#cfcfcf'; }}
    >
      <div style={statCardStyles.label}>{label}</div>
      <div style={statCardStyles.valueRow}>
        <span style={statCardStyles.value}>{value}</span>
        {unit && <span style={statCardStyles.unit}>{unit}</span>}
      </div>
      {trend && (
        <div style={{ ...statCardStyles.trend, color: trendColor }}>
          {trendArrow} {trend}
        </div>
      )}
      {target && (
        <div style={statCardStyles.target}>Target: {target}</div>
      )}
    </div>
  );
};

const statCardStyles = {
  card: {
    background: '#fff',
    border: '1px solid #cfcfcf',
    borderRadius: 6,
    padding: '20px 24px',
    transition: 'border-color 0.15s',
  },
  label: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 13,
    fontWeight: 600,
    color: '#a1a1a1',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: 8,
  },
  valueRow: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 4,
    marginBottom: 6,
  },
  value: {
    fontFamily: "'Montserrat', sans-serif",
    fontSize: 36,
    fontWeight: 700,
    color: '#0f4d90',
    lineHeight: 1,
  },
  unit: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 16,
    fontWeight: 600,
    color: '#444',
  },
  trend: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 4,
  },
  target: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 12,
    color: '#a1a1a1',
    marginTop: 4,
  },
};

Object.assign(window, { StatCard });

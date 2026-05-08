// Callout.jsx — Philly signature callout pattern

const Callout = ({ children, variant, title }) => {
  const borderColors = {
    default: '#f3c613',
    info:    '#2176d2',
    success: '#58c04d',
    warning: '#f99300',
    error:   '#cc3000',
  };
  const borderColor = borderColors[variant || 'default'];

  return (
    <div style={{ ...calloutStyles.base, borderLeftColor: borderColor }}>
      {title && <div style={calloutStyles.title}>{title}</div>}
      <div style={calloutStyles.body}>{children}</div>
    </div>
  );
};

const calloutStyles = {
  base: {
    background: '#f0f0f0',
    borderLeft: '4px solid #f3c613',
    padding: '14px 16px',
  },
  title: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 13,
    fontWeight: 600,
    color: '#444',
    marginBottom: 4,
  },
  body: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 14,
    color: '#444',
    lineHeight: 1.5,
  },
};

Object.assign(window, { Callout });

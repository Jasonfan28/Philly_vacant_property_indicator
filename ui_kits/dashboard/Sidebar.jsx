// Sidebar.jsx — Department filter sidebar

const DEPARTMENTS = [
  { id: 'all',        label: 'All departments' },
  { id: 'streets',    label: 'Streets' },
  { id: 'li',         label: 'Licenses & Inspections' },
  { id: 'ppd',        label: 'Police (PPD)' },
  { id: 'parks',      label: 'Parks & Recreation' },
  { id: 'water',      label: 'Water' },
  { id: 'oit',        label: 'Office of Innovation' },
  { id: 'health',     label: 'Public Health' },
];

const Sidebar = ({ selected, onSelect }) => {
  return (
    <aside style={sidebarStyles.aside}>
      <div style={sidebarStyles.sectionLabel}>Filter by department</div>
      <ul style={sidebarStyles.list}>
        {DEPARTMENTS.map(dept => (
          <li key={dept.id}>
            <button
              onClick={() => onSelect(dept.id)}
              style={{
                ...sidebarStyles.item,
                background: selected === dept.id ? '#daedfe' : 'transparent',
                color: selected === dept.id ? '#0f4d90' : '#444',
                fontWeight: selected === dept.id ? 600 : 400,
                borderLeft: selected === dept.id ? '3px solid #2176d2' : '3px solid transparent',
              }}
            >
              {dept.label}
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
};

const sidebarStyles = {
  aside: {
    width: 220,
    flexShrink: 0,
    borderRight: '1px solid #cfcfcf',
    paddingTop: 24,
    paddingRight: 0,
    minHeight: '100%',
  },
  sectionLabel: {
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 11,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: '#a1a1a1',
    padding: '0 16px',
    marginBottom: 8,
  },
  list: {
    listStyle: 'none',
    margin: 0,
    padding: 0,
  },
  item: {
    display: 'block',
    width: '100%',
    textAlign: 'left',
    padding: '10px 16px',
    fontFamily: "'Open Sans', sans-serif",
    fontSize: 14,
    border: 'none',
    borderLeft: '3px solid transparent',
    cursor: 'pointer',
    transition: 'background 0.1s',
    background: 'transparent',
  },
};

Object.assign(window, { Sidebar, DEPARTMENTS });

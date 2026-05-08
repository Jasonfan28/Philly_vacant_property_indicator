// DataTable.jsx — Sortable civic data table

const MOCK_DATA = [
  { dept: 'Streets',             metric: 'Trash collection on-time rate',     target: '90%',    actual: '94%',      trend: '+2%',  dir: 'up',   status: 'on-track' },
  { dept: 'Licenses & Insp.',    metric: 'Permit processing time',             target: '5 days', actual: '3.8 days', trend: '+18%', dir: 'up',   status: 'on-track' },
  { dept: 'Police (PPD)',        metric: '911 average answer time',            target: '10 sec', actual: '12.4 sec', trend: '-8%',  dir: 'down', status: 'off-target' },
  { dept: 'Parks & Rec',        metric: 'Facility inspection completion',      target: '100%',   actual: '87%',      trend: '-5%',  dir: 'down', status: 'at-risk' },
  { dept: 'Water',               metric: 'Main break repair time',             target: '24 hrs', actual: '19.2 hrs', trend: '+12%', dir: 'up',   status: 'on-track' },
  { dept: 'Office of Innovation','metric': '311 ticket resolution rate',       target: '85%',    actual: '81%',      trend: '-3%',  dir: 'down', status: 'at-risk' },
  { dept: 'Public Health',       metric: 'Restaurant inspection completion',   target: '95%',    actual: '97%',      trend: '+1%',  dir: 'up',   status: 'on-track' },
];

const STATUS_CONFIG = {
  'on-track':  { label: 'On track',  color: '#3a833c' },
  'at-risk':   { label: 'At risk',   color: '#f99300' },
  'off-target':{ label: 'Off target', color: '#cc3000' },
};

const DataTable = ({ filterDept }) => {
  const [sortCol, setSortCol] = React.useState('dept');
  const [sortDir, setSortDir] = React.useState('asc');

  const filtered = filterDept && filterDept !== 'all'
    ? MOCK_DATA.filter(r => r.dept.toLowerCase().includes(filterDept))
    : MOCK_DATA;

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortCol] || '';
    const bv = b[sortCol] || '';
    return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
  });

  const handleSort = col => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('asc'); }
  };

  const thStyle = col => ({
    ...dtStyles.th,
    cursor: 'pointer',
    background: '#0f4d90',
  });

  const chevron = col => sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '';

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={dtStyles.table}>
        <thead>
          <tr>
            {[['dept','Department'],['metric','Metric'],['target','Target'],['actual','Actual'],['trend','Trend'],['status','Status']].map(([col, label]) => (
              <th key={col} style={{ ...thStyle(col), textAlign: col === 'target' || col === 'actual' || col === 'trend' ? 'right' : 'left' }}
                  onClick={() => handleSort(col)}>
                {label}{chevron(col)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const sc = STATUS_CONFIG[row.status];
            const trendColor = row.dir === 'up' ? '#3a833c' : '#cc3000';
            return (
              <tr key={i} style={{ background: i % 2 === 0 ? '#f7f7f7' : '#fff' }}>
                <td style={dtStyles.td}>{row.dept}</td>
                <td style={dtStyles.td}>{row.metric}</td>
                <td style={{ ...dtStyles.td, textAlign: 'right' }}>{row.target}</td>
                <td style={{ ...dtStyles.td, textAlign: 'right', fontWeight: 600 }}>{row.actual}</td>
                <td style={{ ...dtStyles.td, textAlign: 'right', color: trendColor, fontWeight: 600 }}>{row.trend}</td>
                <td style={dtStyles.td}><span style={{ color: sc.color, fontWeight: 600 }}>{sc.label}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

const dtStyles = {
  table: { width: '100%', borderCollapse: 'collapse', fontFamily: "'Open Sans', sans-serif" },
  th: {
    background: '#0f4d90',
    color: '#fff',
    fontFamily: "'Montserrat', sans-serif",
    fontSize: 12,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    padding: '12px 16px',
    userSelect: 'none',
  },
  td: {
    fontSize: 14,
    padding: '12px 16px',
    borderBottom: '1px solid #f0f0f0',
    color: '#444',
  },
};

Object.assign(window, { DataTable, MOCK_DATA });

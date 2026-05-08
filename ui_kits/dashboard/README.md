# Philly Stat 360 — Dashboard UI Kit

## Overview
High-fidelity recreation of the Philly Stat 360 public performance dashboard.
Based on City of Philadelphia Digital Standards; no Figma or codebase provided —
derived from standards.phila.gov and live reference at philly-stat-360.phila.gov.

## Screens
- **Dashboard home** — KPI stat cards + department filter + data table overview
- **Department detail** — Single department deep-dive with chart + table + callouts
- **About / methodology** — Text-heavy page with callout patterns

## Components
| File | Description |
|---|---|
| `Header.jsx` | Top navigation bar with logo + nav links |
| `StatCard.jsx` | KPI stat card with value, label, trend, status |
| `DataTable.jsx` | Sortable civic data table |
| `Callout.jsx` | Philly signature callout (4px yellow left border) |
| `Sidebar.jsx` | Department filter sidebar |

## Usage
Open `index.html` for a click-through prototype of the dashboard.
All components use CSS variables from `../../colors_and_type.css`.

## Notes
- No drop shadows — borders only
- Lucide icons via CDN for UI icons
- Open Sans (Google Fonts) for body; Montserrat (local TTF) for headings

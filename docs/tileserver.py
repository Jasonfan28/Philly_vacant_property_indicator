from flask import Flask, Response, request, jsonify
import psycopg2

app = Flask(__name__)

DB = "postgresql://postgres:291520@localhost/vacancy_dashboard"

# All fields present in the new GeoJSON
ALL_FIELDS = [
    'parcel_number', 'ovs', 'data_split',
    'ensemble_prob', 'ensemble_prob_raw', 'risk_score', 'qtile_tier', 'ensemble_flag',
    'rf_prob', 'logit_prob', 'xgb_prob', 'lgb_prob',
    'rf_flag', 'logit_flag', 'xgb_flag', 'lgb_flag',
    'zip_code', 'census_tract', 'geographic_ward'
]

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# ── Summary stats ──────────────────────────────────────────────────────────────
@app.route("/summary")
def summary():
    try:
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*)                                                                    AS total_parcels,
                SUM(CASE WHEN qtile_tier = 'Top 1% (highest risk)'    THEN 1 ELSE 0 END)  AS top1,
                SUM(CASE WHEN qtile_tier = '95\u201399%'                   THEN 1 ELSE 0 END)  AS p95_99,
                SUM(CASE WHEN qtile_tier = '90\u201395%'                   THEN 1 ELSE 0 END)  AS p90_95,
                SUM(CASE WHEN qtile_tier = '75\u201390%'                   THEN 1 ELSE 0 END)  AS p75_90,
                SUM(CASE WHEN qtile_tier = 'Bottom 75% (lowest risk)' THEN 1 ELSE 0 END)  AS bottom75,
                ROUND(AVG(ensemble_prob)::numeric, 4)                                      AS avg_prob,
                SUM(CASE WHEN ovs = 1 THEN 1 ELSE 0 END)                                  AS observed_vacant,
                SUM(ensemble_flag)                                                         AS total_flagged
            FROM vacancy_predictions
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({
            'total_parcels':   row[0],
            'top1':            row[1],
            'p95_99':          row[2],
            'p90_95':          row[3],
            'p75_90':          row[4],
            'bottom75':        row[5],
            'avg_prob':        float(row[6]) if row[6] else 0,
            'observed_vacant': row[7],
            'total_flagged':   row[8]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Parcel search ──────────────────────────────────────────────────────────────
@app.route("/search")
def search():
    try:
        q = request.args.get('q', '').strip()
        if not q:
            return jsonify([])
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT parcel_number, ensemble_prob, qtile_tier, ensemble_flag, risk_score,
                   ovs, zip_code, census_tract, geographic_ward,
                   rf_prob, logit_prob, xgb_prob, lgb_prob,
                   rf_flag, logit_flag, xgb_flag, lgb_flag,
                   ST_Y(ST_Centroid(wkb_geometry)) AS lat,
                   ST_X(ST_Centroid(wkb_geometry)) AS lng
            FROM vacancy_predictions
            WHERE parcel_number ILIKE %s OR address ILIKE %s
            ORDER BY ensemble_prob DESC
            LIMIT 5
        """, (f'%{q}%', f'%{q}%'))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        keys = ['parcel_number','ensemble_prob','qtile_tier','ensemble_flag','risk_score',
                'ovs','zip_code','census_tract','geographic_ward',
                'rf_prob','logit_prob','xgb_prob','lgb_prob',
                'rf_flag','logit_flag','xgb_flag','lgb_flag',
                'lat','lng']
        return jsonify([dict(zip(keys, r)) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Ward list ──────────────────────────────────────────────────────────────────
@app.route("/wards")
def wards():
    try:
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT geographic_ward::int, COUNT(*) AS total,
                   SUM(ensemble_flag) AS flagged,
                   SUM(ovs) AS observed_vacant
            FROM vacancy_predictions
            WHERE geographic_ward IS NOT NULL
            GROUP BY geographic_ward::int
            ORDER BY geographic_ward::int
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([{
            'ward': r[0], 'total': r[1],
            'flagged': int(r[2]) if r[2] else 0,
            'observed_vacant': int(r[3]) if r[3] else 0
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Ward bounds (for map fly-to) ───────────────────────────────────────────────
@app.route("/ward_bounds")
def ward_bounds():
    try:
        ward = request.args.get('ward', '')
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                ST_YMin(ST_Extent(wkb_geometry)),
                ST_XMin(ST_Extent(wkb_geometry)),
                ST_YMax(ST_Extent(wkb_geometry)),
                ST_XMax(ST_Extent(wkb_geometry))
            FROM vacancy_predictions
            WHERE geographic_ward::int = %s
        """, (int(ward),))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or row[0] is None:
            return jsonify({})
        return jsonify({
            'minlat': row[0], 'minlng': row[1],
            'maxlat': row[2], 'maxlng': row[3]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Ward stats ─────────────────────────────────────────────────────────────────
@app.route("/ward_stats")
def ward_stats():
    try:
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                geographic_ward::int                               AS ward,
                COUNT(*)                                           AS total,
                SUM(ensemble_flag)                                 AS flagged,
                ROUND(AVG(ensemble_prob)::numeric, 4)              AS mean_prob,
                SUM(CASE WHEN ovs = 1 THEN 1 ELSE 0 END)          AS observed_vacant,
                SUM(CASE WHEN qtile_tier = 'Top 1% (highest risk)' THEN 1 ELSE 0 END) AS top1_count
            FROM vacancy_predictions
            WHERE geographic_ward IS NOT NULL
            GROUP BY geographic_ward::int
            ORDER BY flagged DESC NULLS LAST
            LIMIT 10
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([{
            'ward':            r[0],
            'total':           int(r[1]),
            'flagged':         int(r[2]) if r[2] else 0,
            'mean_prob':       float(r[3]) if r[3] else 0,
            'observed_vacant': int(r[4]) if r[4] else 0,
            'top1_count':      int(r[5]) if r[5] else 0,
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Vector tiles: all parcels ──────────────────────────────────────────────────
@app.route("/tiles/vacancy_predictions/<int:z>/<int:x>/<int:y>.pbf")
def tile(z, x, y):
    try:
        ward = request.args.get('ward')
        conn = psycopg2.connect(DB)
        cur = conn.cursor()

        where = "wkb_geometry && ST_Transform(ST_TileEnvelope(%s, %s, %s), 4326)"
        params = [z, x, y, z, x, y]
        if ward:
            where += " AND geographic_ward::int = %s"
            params.append(int(ward))
        census_tract = request.args.get('census_tract')
        if census_tract:
            where += " AND census_tract::numeric::int = %s"
            params.append(int(float(census_tract)))

        cur.execute(f"""
            SELECT ST_AsMVT(q, 'vacancy_predictions', 4096, 'geom')
            FROM (
                SELECT
                    parcel_number, ovs, data_split,
                    ensemble_prob, ensemble_prob_raw, risk_score, qtile_tier, ensemble_flag,
                    rf_prob, logit_prob, xgb_prob, lgb_prob,
                    rf_flag, logit_flag, xgb_flag, lgb_flag,
                    zip_code, census_tract, geographic_ward,
                    ST_AsMVTGeom(
                        ST_Transform(wkb_geometry, 3857),
                        ST_TileEnvelope(%s, %s, %s),
                        4096, 64, true
                    ) AS geom
                FROM vacancy_predictions
                WHERE {where}
            ) q
            WHERE geom IS NOT NULL
        """, params)

        tile_data = cur.fetchone()[0]
        cur.close()
        conn.close()
        return Response(bytes(tile_data), mimetype='application/x-protobuf',
                        headers={'Access-Control-Allow-Origin': '*'})
    except Exception as e:
        return Response(status=500)


# ── Vector tiles: flagged parcels only ────────────────────────────────────────
@app.route("/tiles/vacancy_predictions_flagged/<int:z>/<int:x>/<int:y>.pbf")
def tile_flagged(z, x, y):
    try:
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT ST_AsMVT(q, 'vacancy_predictions_flagged', 4096, 'geom')
            FROM (
                SELECT
                    parcel_number, ovs, data_split,
                    ensemble_prob, ensemble_prob_raw, risk_score, qtile_tier, ensemble_flag,
                    rf_prob, logit_prob, xgb_prob, lgb_prob,
                    rf_flag, logit_flag, xgb_flag, lgb_flag,
                    zip_code, census_tract, geographic_ward,
                    ST_AsMVTGeom(
                        ST_Transform(wkb_geometry, 3857),
                        ST_TileEnvelope(%s, %s, %s),
                        4096, 64, true
                    ) AS geom
                FROM vacancy_predictions_flagged
                WHERE wkb_geometry && ST_Transform(ST_TileEnvelope(%s, %s, %s), 4326)
            ) q
            WHERE geom IS NOT NULL
        """, (z, x, y, z, x, y))
        tile_data = cur.fetchone()[0]
        cur.close()
        conn.close()
        return Response(bytes(tile_data), mimetype='application/x-protobuf',
                        headers={'Access-Control-Allow-Origin': '*'})
    except Exception as e:
        return Response(status=500)


# ── Point query (map click) ────────────────────────────────────────────────────
@app.route("/query")
def query():
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT parcel_number, ovs, data_split,
                   ensemble_prob, ensemble_prob_raw, risk_score, qtile_tier, ensemble_flag,
                   rf_prob, logit_prob, xgb_prob, lgb_prob,
                   rf_flag, logit_flag, xgb_flag, lgb_flag,
                   zip_code, census_tract, geographic_ward
            FROM vacancy_predictions
            WHERE ST_Contains(wkb_geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
            LIMIT 1
        """, (lng, lat))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return jsonify({})
        return jsonify(dict(zip(ALL_FIELDS, row)))
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ── Census tract list ──────────────────────────────────────────────────────────
@app.route("/census_tracts")
def census_tracts():
    try:
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT census_tract::numeric::int, COUNT(*) AS total,
                   SUM(ensemble_flag) AS flagged
            FROM vacancy_predictions
            WHERE census_tract IS NOT NULL
            GROUP BY census_tract::numeric::int
            ORDER BY census_tract::numeric::int
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([{'tract': r[0], 'total': r[1], 'flagged': int(r[2]) if r[2] else 0} for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Census tract bounds (for map fly-to) ───────────────────────────────────────
@app.route("/tract_bounds")
def tract_bounds():
    try:
        tract = request.args.get('census_tract', '')
        conn = psycopg2.connect(DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                ST_YMin(ST_Extent(wkb_geometry)),
                ST_XMin(ST_Extent(wkb_geometry)),
                ST_YMax(ST_Extent(wkb_geometry)),
                ST_XMax(ST_Extent(wkb_geometry))
            FROM vacancy_predictions
            WHERE census_tract::numeric::int = %s
        """, (int(float(tract)),))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row or row[0] is None:
            return jsonify({})
        return jsonify({
            'minlat': row[0], 'minlng': row[1],
            'maxlat': row[2], 'maxlng': row[3]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Tile server running at http://localhost:5001")
    app.run(port=5001, debug=False)

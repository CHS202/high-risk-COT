WITH extubation_cases AS (
  SELECT stay_id, hadm_id, extubation_time, outcome
  FROM `physionet-data-452606.finalproject.extubation_cases` -- Your custom cohort table
),
mode_intervals AS (
  -- Identifies time intervals for each ventilator mode
  SELECT
    ce.stay_id,
    ce.charttime AS start_time,
    LEAD(ce.charttime) OVER (PARTITION BY ce.stay_id ORDER BY ce.charttime) AS end_time,
    ce.value AS mode
  FROM
    `physionet-data.mimiciv_2_2_icu.chartevents` ce -- Make sure schema name is correct (e.g., mimiciv_icu)
  JOIN
    extubation_cases ec ON ce.stay_id = ec.stay_id
  WHERE
    ce.itemid = 223849 -- Ventilator mode
),
rr_spontaneous AS (
  -- Selects spontaneous respiratory rate during appropriate modes and time window
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS rr,
    ec.extubation_time -- Carry extubation_time forward for filtering
  FROM
    `physionet-data.mimiciv_2_2_icu.chartevents` ce
  JOIN
    extubation_cases ec ON ce.stay_id = ec.stay_id
  JOIN
    mode_intervals mi ON ce.stay_id = mi.stay_id
    AND ce.charttime >= mi.start_time
    AND (ce.charttime < mi.end_time OR mi.end_time IS NULL)
  WHERE
    ce.itemid = 224688 -- RR (spontaneous)
    AND (LOWER(mi.mode) LIKE '%cpap%' OR LOWER(mi.mode) LIKE '%psv%' OR LOWER(mi.mode) LIKE '%sbt%') -- Use LOWER for case-insensitivity
    --AND mi.mode LIKE 'PSV/SBT'
    --AND ce.charttime BETWEEN DATETIME_SUB(ec.extubation_time, INTERVAL 3 DAY) AND ec.extubation_time
    AND ce.charttime < ec.extubation_time
    AND ce.valuenum > 0 --AND ce.valuenum < 100 -- Basic sanity check for RR
),
vt_spontaneous AS (
  -- Selects spontaneous tidal volume during appropriate modes and time window
  SELECT
    ce.stay_id,
    ce.charttime,
    ce.valuenum AS vt,
    ec.extubation_time -- Carry extubation_time forward for filtering
  FROM
    `physionet-data.mimiciv_2_2_icu.chartevents` ce
  JOIN
    extubation_cases ec ON ce.stay_id = ec.stay_id
  JOIN
    mode_intervals mi ON ce.stay_id = mi.stay_id
    AND ce.charttime >= mi.start_time
    AND (ce.charttime < mi.end_time OR mi.end_time IS NULL)
  WHERE
    ce.itemid = 224685 -- VT (Spontaneous)
    AND (LOWER(mi.mode) LIKE '%cpap%' OR LOWER(mi.mode) LIKE '%psv%' OR LOWER(mi.mode) LIKE '%sbt%') -- Use LOWER for case-insensitivity
    --AND mi.mode LIKE 'PSV/SBT'
    --AND ce.charttime BETWEEN DATETIME_SUB(ec.extubation_time, INTERVAL 3 DAY) AND ec.extubation_time
    AND ce.charttime < ec.extubation_time
    AND ce.valuenum > 0 --AND ce.valuenum < 2000 -- Basic sanity check for VT (mL)
),
rsbi_pairs AS (
  -- Pairs RR and VT measurements that are close in time and calculates RSBI
  SELECT
    rr.stay_id,
    rr.charttime AS rr_time,
    vt.charttime AS vt_time,
    rr.rr,
    vt.vt,
    (rr.rr / (vt.vt / 1000.0)) AS rsbi,
    -- Rank VT measurements by proximity to the RR measurement time
    ROW_NUMBER() OVER (PARTITION BY rr.stay_id, rr.charttime ORDER BY ABS(TIMESTAMP_DIFF(rr.charttime, vt.charttime, SECOND))) AS rn_time_diff
  FROM
    rr_spontaneous rr
  JOIN
    vt_spontaneous vt ON rr.stay_id = vt.stay_id
  -- WHERE
    -- *** ADDED TIME DIFFERENCE CONSTRAINT: Only pair measurements within 10 minutes ***
    -- ABS(TIMESTAMP_DIFF(rr.charttime, vt.charttime, MINUTE)) <= 10
    -- vt.vt > 0 -- This is redundant due to filter in vt_spontaneous CTE
),
latest_rsbi AS (
  -- Selects the closest VT for each RR and then finds the latest RSBI value per stay
  SELECT
    stay_id,
    rr_time,
    rr,
    vt,
    rsbi,
    -- Rank the RSBI calculations by time (latest first) for each patient
    ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY rr_time DESC) AS rn_latest
  FROM
    rsbi_pairs
  WHERE
    rn_time_diff = 1 -- Only keep the VT measurement closest in time to the RR measurement
)
-- Final selection of the single latest RSBI value per patient
SELECT
  stay_id,
  rr_time AS charttime, -- Represents the timestamp of the RR measurement used for the latest RSBI
  rr,
  vt, -- The corresponding VT value (closest in time within the limit)
  rsbi
FROM
  latest_rsbi
WHERE
  rn_latest = 1 -- Filter to get only the latest RSBI calculation per stay_id
ORDER BY
  stay_id
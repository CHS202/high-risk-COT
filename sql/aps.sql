WITH extubation_cases AS (
  SELECT
    *
  FROM `nthu-mimic.final_project.cohort`
),

vital_data AS (
  SELECT
    chart.stay_id,
    chart.charttime,
    chart.itemid,
    chart.valuenum,
    e.extubation_time,
    ROW_NUMBER() OVER (
      PARTITION BY chart.stay_id, chart.itemid
      ORDER BY chart.charttime DESC
    ) AS rn
  FROM extubation_cases e
  JOIN `physionet-data.mimiciv_2_2_icu.chartevents` chart
    ON e.stay_id = chart.stay_id
  WHERE 
    chart.itemid IN (
        223762, 223761, -- Temperature Celsius, Temperature Fahrenheit
        220045, -- Heart rate
        220210, -- Respiratory rate
        220052, 220181,  -- Arterial Blood Pressure mean
        220545 -- Hematocrit (serum)
    )
    AND 
      CAST(chart.charttime AS TIMESTAMP)
      BETWEEN TIMESTAMP_SUB(e.extubation_time, INTERVAL 3 DAY)
      AND e.extubation_time
),

vital_table AS (
  SELECT 
    stay_id,
    Round(
      MAX(
      CASE WHEN itemid = 223762 THEN valuenum
        WHEN itemid = 223761 THEN (valuenum - 32) / 1.8
        END
        )
      , 2) AS temperature,
    MAX(
      CASE WHEN itemid = 220052 THEN valuenum 
      WHEN itemid = 220181 THEN valuenum
      END) AS mean_art_pressure,
    MAX(CASE WHEN itemid = 220045 THEN valuenum END) AS heart_rate,
    MAX(CASE WHEN itemid = 220210 THEN valuenum END) AS resp_rate,
    MAX(CASE WHEN itemid = 220545 THEN valuenum END) AS hematocrit
  FROM vital_data
  WHERE rn = 1
  GROUP BY stay_id
),

blood_gas AS (
  SELECT 
    hadm_id,
    -- 5. Oxygenation
    IFNULL(
      fio2
      , fio2_chartevents 
    )/100 as fio2_5,
    -- Note: fio2_chartevents is less likely to miss value than bg.fio2
    COALESCE(aado2, aado2_calc)/100 as aado2_5,
    pao2fio2ratio * (COALESCE(aado2, aado2_calc)/100) as pao2_5,
    -- 6. Arterial pH
    ph AS arterial_ph_6,
    -- 10. Hematocrit (%)
    hematocrit AS heratocrit_10,
    pao2fio2ratio
  FROM (
    SELECT 
      bg.hadm_id, 
      bg.pco2,
      bg.fio2,
      bg.fio2_chartevents,
      bg.aado2,
      bg.aado2_calc,
      bg.po2, 
      bg.pao2fio2ratio,
      bg.ph,
      bg.hematocrit,
      ROW_NUMBER() OVER (PARTITION BY bg.hadm_id ORDER BY bg.charttime DESC) AS rn
    FROM extubation_cases e
    LEFT JOIN `physionet-data.mimiciv_2_2_derived.bg` bg
    ON bg.hadm_id = e.hadm_id
    WHERE 
      bg.specimen = 'ART.'
      AND CAST(bg.charttime AS TIMESTAMP)
      BETWEEN TIMESTAMP_SUB(e.extubation_time, INTERVAL 3 DAY) AND e.extubation_time
  ) WHERE rn = 1
),

blood_gas_data AS (
    SELECT
    chart.stay_id,
    chart.charttime,
    chart.itemid,
    chart.valuenum,
    e.extubation_time,
    ROW_NUMBER() OVER (
      PARTITION BY chart.stay_id, chart.itemid
      ORDER BY chart.charttime DESC
    ) AS rn
  FROM extubation_cases e
  JOIN `physionet-data.mimiciv_2_2_icu.chartevents` chart
    ON e.stay_id = chart.stay_id
  WHERE 
    chart.itemid IN (
      220545 -- Hematocrit (serum)
    )
    AND 
      CAST(chart.charttime AS TIMESTAMP)
      BETWEEN TIMESTAMP_SUB(e.extubation_time, INTERVAL 3 DAY)
      AND e.extubation_time
),


chemistry AS (
  SELECT
    hadm_id,
    -- 7. Serum sodium (mmol/L or mEq/L)
    sodium AS serum_sodium_7,
    -- 8. Serum potassium (mmol/L or mEq/L)
    potassium AS serum_potassium_8,
    -- 9. Serum creatinine (micromol/L or mg/dL); 
    -- double point score for patients with acute renal failure
    creatinine AS serum_creatinine_9,
  FROM (
    SELECT
      e.hadm_id,
      sodium,
      potassium,
      creatinine,
      ROW_NUMBER() OVER (PARTITION BY chem.hadm_id ORDER BY chem.charttime DESC) AS rn
    FROM extubation_cases e
    LEFT JOIN `physionet-data.mimiciv_2_2_derived.chemistry` chem
    ON e.hadm_id = chem.hadm_id
    WHERE 
      CAST(chem.charttime AS TIMESTAMP)
      BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) 
      AND e.extubation_time
  )
  WHERE
    rn = 1
),

blood AS (
  SELECT
    hadm_id,
    -- 11. White blood cells (in 1000s)
    wbc AS wbc_11
  FROM (
    SELECT
      e.hadm_id,
      blood.wbc,
      ROW_NUMBER() OVER (PARTITION BY blood.hadm_id ORDER BY blood.charttime DESC) AS rn
    FROM extubation_cases e
    LEFT JOIN `physionet-data.mimiciv_2_2_derived.complete_blood_count` blood
    ON e.hadm_id = blood.hadm_id
    WHERE 
      CAST(blood.charttime AS TIMESTAMP)
      BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) 
      AND e.extubation_time
  )
  WHERE
    rn = 1
),

gcs AS (
  SELECT
    hadm_id,
    -- 12. Glasgow coma score (GCS)
    gcs AS glasgow_coma_score_12,  --TODO: fix this gcs_min maybe incorrect
  FROM (
    SELECT
      e.hadm_id,
      gcs.gcs,
      ROW_NUMBER() OVER (PARTITION BY gcs.stay_id ORDER BY gcs.charttime DESC) AS rn
    FROM extubation_cases e
    LEFT JOIN `physionet-data.mimiciv_2_2_derived.gcs` gcs
    ON e.stay_id = gcs.stay_id
    WHERE 
      CAST(gcs.charttime AS TIMESTAMP)
      BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) 
      AND e.extubation_time
  )
  WHERE
    rn = 1
)

SELECT
  e.stay_id,
  e.hadm_id,
  -- 1. Temperature, core (°C)
  vital.temperature AS temp_core_1,
  -- 2. Mean arterial pressure (mm Hg)
  vital.mean_art_pressure as mean_art_pressure_2,
  -- 3. Heart rate
  vital.heart_rate as heart_rate_3,
  -- 4. Respiratory rate (nonventilated or ventilated)
  vital.resp_rate as resp_rate_4,
  -- 5. Oxygenation
  bg.fio2_5,
  bg.aado2_5,
  bg.pao2_5,
  bg.arterial_ph_6,
  chem.serum_sodium_7,
  chem.serum_potassium_8,
  chem.serum_creatinine_9,
  vital.hematocrit AS heratocrit_10,
  blood.wbc_11,
  gcs.glasgow_coma_score_12,
  -- Other features
  bg.pao2fio2ratio
FROM extubation_cases e
LEFT JOIN vital_table vital
ON e.stay_id = vital.stay_id
LEFT JOIN blood_gas bg
ON e.hadm_id = bg.hadm_id
LEFT JOIN chemistry chem
ON e.hadm_id = chem.hadm_id
LEFT JOIN blood
ON e.hadm_id = blood.hadm_id
LEFT JOIN gcs
ON e.hadm_id = gcs.hadm_id

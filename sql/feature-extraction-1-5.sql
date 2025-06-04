WITH extubation_cases AS (
  SELECT stay_id, hadm_id, extubation_time, outcome
  FROM `physionet-data-452606.finalproject.extubation_cases`
),

-- 1. Disease severity (APACHE Score) - Using APS III as a proxy
apache_iii AS (
  SELECT stay_id, apache_iii_score
  FROM `physionet-data-452606.finalproject.apache_iii`
  WHERE stay_id IN (SELECT stay_id FROM extubation_cases)
),

-- 2. Lab data (blood gas: CO2, O2) and 5. PaO2 / FiO2 ratio
blood_gas AS (
  SELECT hadm_id, pco2, po2, pao2fio2ratio
  FROM (
    SELECT bg.hadm_id, bg.pco2, bg.po2, bg.pao2fio2ratio,
           ROW_NUMBER() OVER (PARTITION BY bg.hadm_id ORDER BY bg.charttime DESC) AS rn
    FROM `physionet-data.mimiciv_2_2_derived.bg` bg
    JOIN extubation_cases e ON bg.hadm_id = e.hadm_id
    WHERE bg.charttime BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) AND e.extubation_time
      AND bg.specimen = 'ART.'
  ) WHERE rn = 1
),

-- 3. Weaning parameters (Pi/e max - using negative respiratory force)
pi_max_recent AS (
  SELECT stay_id, value AS pi_max
  FROM (
    SELECT c.stay_id, c.value,
           ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY c.charttime DESC) AS rn
    FROM `physionet-data.mimiciv_2_2_icu.chartevents` c
    JOIN extubation_cases e ON c.stay_id = e.stay_id
    WHERE c.itemid = 224419  -- negative respiratory force
      AND c.charttime BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) AND e.extubation_time
  ) WHERE rn = 1
),
-- 3. rapid shallow breathing index (RSBI) use no limit time before extubation
rsbi AS (
  SELECT stay_id, rsbi
  FROM `physionet-data-452606.finalproject.rsbi-all`
  WHERE stay_id IN (SELECT stay_id FROM extubation_cases)
),
-- 6. Lower hemoglobin level
hemoglobin_recent AS (
  SELECT hadm_id, hemoglobin
  FROM (
    SELECT h.hadm_id, h.hemoglobin,
           ROW_NUMBER() OVER (PARTITION BY h.hadm_id ORDER BY h.charttime DESC) AS rn
    FROM `physionet-data.mimiciv_2_2_derived.complete_blood_count` h
    JOIN extubation_cases e ON h.hadm_id = e.hadm_id
    WHERE h.charttime BETWEEN DATETIME_SUB(e.extubation_time, INTERVAL 3 DAY) AND e.extubation_time
  ) WHERE rn = 1
)

-- Combine all features
SELECT
  e.stay_id,
  e.outcome,
  a.apache_iii_score AS apache_score,
  b.pco2 AS blood_gas_co2,
  b.po2 AS blood_gas_o2,
  p.pi_max AS pi_max,
  r.rsbi AS rsbi,
  b.pao2fio2ratio AS pao2_fio2_ratio,
  h.hemoglobin AS hemoglobin,
FROM extubation_cases e
LEFT JOIN apache_iii a ON e.stay_id = a.stay_id
LEFT JOIN blood_gas b ON e.hadm_id = b.hadm_id
LEFT JOIN pi_max_recent p ON e.stay_id = p.stay_id
LEFT JOIN rsbi r ON e.stay_id = r.stay_id
LEFT JOIN hemoglobin_recent h ON e.hadm_id = h.hadm_id
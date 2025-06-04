WITH icu_info AS (
  SELECT
    detail.subject_id,
    detail.stay_id,
    hadm_id,
    admission_age AS age,
    admittime AS admission_time,
    icu_intime,
    icu_outtime,
    TIMESTAMP_DIFF(icu_outtime, icu_intime, HOUR) AS los_icu_hr,
    first_icu_stay,
    weight_table.weight_admit AS adm_weight_kg,
    height_table.height AS adm_height_cm,
    ROUND((weight_admit / (height * height)) * 10000, 2) AS adm_BMI
  FROM `physionet-data.mimiciv_derived.icustay_detail` detail
  LEFT JOIN `physionet-data.mimiciv_derived.first_day_weight` weight_table
  ON detail.stay_id = weight_table.stay_id
  LEFT JOIN `physionet-data.mimiciv_derived.first_day_height` height_table
  ON detail.stay_id = height_table.stay_id
),

diagnose_table AS (
  SELECT
  hadm_id,
  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "428%" or
        icd_code LIKE "I50%"
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_heart_failure,

  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "414%" or
        icd_code LIKE "I50%"
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_ischemic_heart_disease,

  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "5184%" or
        icd_code LIKE "J81%"
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_pulmonary_edema,

  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "417%" or
        icd_code LIKE "I48%" or
        icd_code LIKE "I49%" 
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_arrhythmia,

  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "496%" or
        icd_code LIKE "J44%" 
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_copd,

  IF(MAX(
    CASE
      WHEN (
        icd_code LIKE "2780%" or
        icd_code LIKE "E66%" 
      ) THEN 1
      ELSE 0
    END
  ) > 0, TRUE, FALSE
  ) AS has_obesity
  FROM
  `physionet-data.mimiciv_2_2_hosp.diagnoses_icd`
  GROUP BY hadm_id
),

chart_event AS (
  SELECT
    stay_id,
    IF( 
      MAX(
        IF (
          itemid IN (
            229150,
            229152,
            229154,
            229155
          )
          , 1
          , 0
        )
      ) > 0
      , TRUE
      , FALSE
    ) AS give_up_treatment,

    IF(
      MAX(
        IF(
          (
            itemid = 223991 -- Cough Effort
            and value = "Weak"
          ) or (
            itemid = 224368 -- Sputum Consistency
            and value IN ("Plug", "Thick", "Tenacious")
          ) or (
            itemid = 224373 -- Sputum Amount
            and value = "Copious"
          )
          , 1
          , 0
        )
      ) > 0
      , TRUE
      , FALSE
    ) AS inadequate_secretions_management
  FROM `physionet-data.mimiciv_2_2_icu.chartevents`
  GROUP By stay_id
),

final_group AS (
  SELECT 
    info.subject_id,
    info.hadm_id,
    info.stay_id,
    info.admission_time,
    info.icu_intime,
    info.icu_outtime,
    info.age,
    info.adm_height_cm,
    info.adm_weight_kg,
    info.adm_BMI,
    info.los_icu_hr,
    info.first_icu_stay,
    dia.has_heart_failure,
    dia.has_ischemic_heart_disease,
    dia.has_pulmonary_edema,
    dia.has_arrhythmia,
    dia.has_copd,
    dia.has_obesity,
    chart.inadequate_secretions_management,
    apa.apache_ii
  FROM icu_info info
  LEFT JOIN chart_event chart
  ON info.stay_id = chart.stay_id
  LEFT JOIN diagnose_table dia
  ON info.hadm_id = dia.hadm_id
  LEFT JOIN `nthu-mimic.1.apache_ii` apa
  ON info.stay_id = apa.stay_id
),

high_risk_table AS (
  SELECT
  *,
  IF( -- High risk paitent
    -- age > 65
    age > 65
    -- APCHE score > 12 on extubation day
    OR apache_ii > 12

    -- BMI > 30 
    OR adm_BMI > 30
    -- Inadequate secretions management
    OR inadequate_secretions_management
    -- -- more than 1 comorbidity
    OR IF (
      (
        IF (has_obesity, 1, 0)
        + IF (has_heart_failure, 1, 0)
        + IF (has_ischemic_heart_disease, 1, 0)
        + IF (has_pulmonary_edema, 1, 0)
        + IF (has_arrhythmia, 1, 0)
        + IF (has_copd, 1, 0)
      ) >= 2
      , TRUE
      , FALSE
    )
  , TRUE
  , FALSE
  ) AS high_risk,
  FROM
    final_group
)

SELECT
  cohort.stay_id,
  cohort.subject_id,
  hr.adm_BMI,
  CAST(hr.inadequate_secretions_management AS INT) AS inadequate_secretions_management,
  CAST(hr.has_obesity AS INT) AS has_obesity,
  CAST(hr.has_heart_failure AS INT) AS has_heart_failure,
  CAST(hr.has_ischemic_heart_disease AS INT) AS has_ischemic_heart_disease,
  CAST(hr.has_pulmonary_edema AS INT) AS has_pulmonary_edema,
  CAST(hr.has_arrhythmia AS INT) AS has_arrhythmia,
  CAST(hr.has_copd AS INT) AS has_copd,
  CAST(hr.high_risk AS INT) AS high_risk
FROM `nthu-mimic.final_project.cohort` cohort
LEFT JOIN high_risk_table hr
ON cohort.stay_id = hr.stay_id

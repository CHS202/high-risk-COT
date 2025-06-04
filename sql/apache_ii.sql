WITH aps AS (
  SELECT *
  FROM `nthu-mimic.final_project.aps`
),

chronic_table AS (
  SELECT
  adm.hadm_id,
  IF(
    MAX(
      IF(icd_code LIKE "I50%", 1, 0)
    ) > 0
    ,1 ,0
  ) AS has_heart_failure,
  IF(
    MAX(
      IF(icd_code LIKE "K74%", 1, 0)
    ) > 0 
    ,1, 0
  ) AS has_cirrhosis,
  IF(
    MAX(
      IF(icd_code LIKE "J44%", 1, 0)
    ) > 0 
    ,1 ,0
  ) AS has_chronic_lung_disese,
  IF (
    MAX(
      IF(icd_code LIKE "N186%", 1, 0)
    ) > 0,
    1, 0
  ) AS has_ESRD,
  CASE
    WHEN
      IF (
        MAX(
          IF (
            icd_code LIKE "I50%"  -- Heart Failure
            or icd_code LIKE "K74%" -- Cirrhosis
            or icd_code LIKE "J44%"  -- Chronic lung disease (COPD proxy)
            or icd_code LIKE "N186%" -- Dialysis-dependent (ESRD)
          , 1
          , 0
          )
        ) > 0,
        TRUE,
        FALSE
      ) 
      AND ANY_VALUE(adm.admission_type) != 'ELECTIVE'
    THEN 5

    WHEN
      IF (
        MAX(
          IF (
            icd_code LIKE "I50%"  -- Heart Failure
            or icd_code LIKE "K74%" -- Cirrhosis
            or icd_code LIKE "J44%"  -- Chronic lung disease (COPD proxy)
            or icd_code LIKE "N186%" -- Dialysis-dependent (ESRD)
          , 1
          , 0
          )
        ) > 0,
        TRUE,
        FALSE
      ) 
      AND ANY_VALUE(adm.admission_type) = 'ELECTIVE'
    THEN 2
    ELSE 0
  END AS chronic_point
  FROM `physionet-data.mimiciv_2_2_hosp.admissions` adm
  LEFT JOIN `physionet-data.mimiciv_2_2_hosp.diagnoses_icd` dia
  ON adm.hadm_id = dia.hadm_id
  GROUP BY hadm_id
),

age_table AS (
  SELECT
  detail.stay_id,
  detail.hadm_id,
  CASE
    WHEN admission_age <= 44
    THEN 0
    WHEN 45 <= admission_age and admission_age <= 54
    THEN 2
    WHEN 55 <= admission_age and admission_age <= 64
    THEN 3
    WHEN 65 <= admission_age and admission_age <= 74
    THEN 5
    WHEN 75 <= admission_age
    THEN 6
    ELSE NULL
  END AS age_point,
  FROM `physionet-data.mimiciv_2_2_derived.icustay_detail` detail
),

aps_table as (
  SELECT
  stay_id,
  hadm_id,
  (
    (
      CASE
        WHEN temp_core_1 >= 41 THEN 4
        WHEN temp_core_1 >= 39 THEN 3
        WHEN temp_core_1 >= 38.5 THEN 1
        WHEN temp_core_1 >= 36 THEN 0
        WHEN temp_core_1 >= 34 THEN 1
        WHEN temp_core_1 >= 32 THEN 2
        WHEN temp_core_1 >= 30 THEN 3
        WHEN temp_core_1 < 30 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN mean_art_pressure_2 >= 160 THEN 4
        WHEN mean_art_pressure_2 >= 130 THEN 3
        WHEN mean_art_pressure_2 >= 110 THEN 2
        WHEN mean_art_pressure_2 >= 70 THEN 0
        WHEN mean_art_pressure_2 >= 50 THEN 2
        WHEN mean_art_pressure_2 < 50 THEN 4
        ELSE null
      END 
    ) + 
    (
      CASE
        WHEN heart_rate_3 >= 180 THEN 4
        WHEN heart_rate_3 >= 140 THEN 3
        WHEN heart_rate_3 >= 110 THEN 2
        WHEN heart_rate_3 >= 70 THEN 0
        WHEN heart_rate_3 >= 55 THEN 2
        WHEN heart_rate_3 >= 40 THEN 3
        WHEN heart_rate_3 < 40 THEN 4
        ELSE null
      END 
    ) + 
    (
      CASE
        WHEN resp_rate_4 >= 50 THEN 4
        WHEN resp_rate_4 >= 35 THEN 3
        WHEN resp_rate_4 >= 25 THEN 1
        WHEN resp_rate_4 >= 12 THEN 0
        WHEN resp_rate_4 >= 10 THEN 1
        WHEN resp_rate_4 >= 6 THEN 2
        WHEN resp_rate_4 < 6 THEN 4
        ELSE null
      END
    ) +
    IF(
      fio2_5 >= 0.5,
      CASE
        WHEN aado2_5 >= 500 THEN 4
        WHEN aado2_5 >= 350 THEN 3
        WHEN aado2_5 >= 200 THEN 2
        WHEN aado2_5 < 200 THEN 0
        ELSE null
      END,
      CASE
        WHEN pao2_5 > 70 THEN 0
        WHEN pao2_5 >= 61 THEN 1
        WHEN pao2_5 >= 55 THEN 3
        WHEN pao2_5 < 55 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN arterial_ph_6 >= 7.7 THEN 4
        WHEN arterial_ph_6 >= 7.6 THEN 3
        WHEN arterial_ph_6 >= 7.5 THEN 1
        WHEN arterial_ph_6 >= 7.33 THEN 0
        WHEN arterial_ph_6 >= 7.25 THEN 2
        WHEN arterial_ph_6 >= 7.15 THEN 3
        WHEN arterial_ph_6 < 7.15 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN serum_sodium_7 >= 180 THEN 4
        WHEN serum_sodium_7 >= 160 THEN 3
        WHEN serum_sodium_7 >= 155 THEN 2
        WHEN serum_sodium_7 >= 150 THEN 1
        WHEN serum_sodium_7 >= 130 THEN 0
        WHEN serum_sodium_7 >= 120 THEN 2
        WHEN serum_sodium_7 >= 111 THEN 3
        WHEN serum_sodium_7 < 111 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN serum_potassium_8 >= 7 THEN 4
        WHEN serum_potassium_8 >= 6 THEN 3
        WHEN serum_potassium_8 >= 5.5 THEN 1
        WHEN serum_potassium_8 >= 3.5 THEN 0
        WHEN serum_potassium_8 >= 3 THEN 1
        WHEN serum_potassium_8 >= 2.5 THEN 2
        WHEN serum_potassium_8 < 2.5 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN serum_creatinine_9 >= 3.5 THEN 4
        WHEN serum_creatinine_9 >= 2 THEN 3
        WHEN serum_creatinine_9 >= 1.5 THEN 2
        WHEN serum_creatinine_9 >= 0.6 THEN 0
        WHEN serum_creatinine_9 < 0.6 THEN 2
        ELSE null
      END
    ) +
    (
      CASE
        WHEN heratocrit_10 >= 60 THEN 4
        WHEN heratocrit_10 >= 50 THEN 2
        WHEN heratocrit_10 >= 46 THEN 1
        WHEN heratocrit_10 >= 30 THEN 0
        WHEN heratocrit_10 >= 20 THEN 2
        WHEN heratocrit_10 < 20 THEN 4
        ELSE null
      END
    ) +
    (
      CASE
        WHEN wbc_11 >= 40 THEN 4
        WHEN wbc_11 >= 20 THEN 2
        WHEN wbc_11 >= 15 THEN 1
        WHEN wbc_11 >= 3 THEN 0
        WHEN wbc_11 >= 1 THEN 2
        WHEN wbc_11 < 1 THEN 4
        ELSE null
      END
    ) +
    (
      15 - glasgow_coma_score_12
    )
  ) AS aps_point,
  FROM aps
)

SELECT
  aps.stay_id,
  aps_point + age_point + chronic_point
  AS apache_ii,
  COALESCE(aps_point, 0) + COALESCE(age_point, 0) + COALESCE(chronic_point, 0)
  AS min_apache_ii,
  chron.has_heart_failure,
  chron.has_cirrhosis,
  chron.has_chronic_lung_disese,
  chron.has_ESRD
FROM aps_table aps
LEFT JOIN age_table age
ON aps.stay_id = age.stay_id
LEFT JOIN chronic_table chron
ON aps.hadm_id = chron.hadm_id
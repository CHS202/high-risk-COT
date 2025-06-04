WITH extubation_cases_base AS (
  -- This CTE selects all columns from your extubation_cases table
  -- It's good practice to list them explicitly if you don't need all of them
  -- or if the table structure might change. For this example, I'll use *
  SELECT *
  FROM `physionet-data-452606.finalproject.extubation_cases` -- Make sure this project and dataset path is correct
)

SELECT
  ec.stay_id,
  -- Patient demographics
  pat.gender,
  -- Admission details
  adm.race

FROM extubation_cases_base AS ec
LEFT JOIN `physionet-data.mimiciv_2_2_hosp.patients` AS pat
  ON ec.subject_id = pat.subject_id
LEFT JOIN `physionet-data.mimiciv_2_2_hosp.admissions` AS adm
  ON ec.hadm_id = adm.hadm_id
LEFT JOIN `physionet-data.mimiciv_2_2_icu.icustays` AS icu
  ON ec.stay_id = icu.stay_id
SELECT
  co.subject_id,
  co.hadm_id,
  co.stay_id,
  co.outcome, 
  -- From original features
  apache_score, 
  blood_gas_co2, 
  blood_gas_o2, 
  pi_max, 
  rsbi, 
  pao2_fio2_ratio, 
  hemoglobin, 
  glasgow_coma_scale, 
  age, 
  has_cardiac_disease, 
  has_respiratory_disease, 
  has_pneumonia, 
  ventilation_duration_hours, 
  heart_rate,
  -- From aps table 
  temp_core_1,
  mean_art_pressure_2, 
  heart_rate_3, 
  resp_rate_4, 
  fio2_5, 
  aado2_5, 
  pao2_5, 
  arterial_ph_6, 
  serum_sodium_7, 
  serum_potassium_8, 
  serum_creatinine_9, 
  heratocrit_10, 
  wbc_11, 
  glasgow_coma_score_12, 
  pao2fio2ratio,
  -- From apach_ii table
  ape.apache_ii,
  ape.min_apache_ii,
  ape.has_chronic_lung_disese,
  ape.has_heart_failure,
  ape.has_cirrhosis,
  ape.has_ESRD,
  -- From high risk table
  adm_BMI, 
  inadequate_secretions_management, 
  has_obesity, 
  has_ischemic_heart_disease, 
  has_pulmonary_edema,
  has_arrhythmia, 
  has_copd,
  high_risk
FROM `nthu-mimic.final_project.cohort` co
JOIN `nthu-mimic.final_project.aps` aps
  ON co.stay_id = aps.stay_id
JOIN `nthu-mimic.final_project.apache_ii` ape
  ON co.stay_id = ape.stay_id
JOIN `nthu-mimic.final_project.high_risk_v1` hr
  ON co.stay_id = hr.stay_id
JOIN `nthu-mimic.final_project.feature_extracted_all` fe
  ON co.stay_id = fe.stay_id

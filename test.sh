echo -e "======== Test 01 ========\n"

curl --silent -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "start_date": "March 1996",
  "complete_date": "July 2000",
  "icdcodes": "D89.810__C95.91",
  "eligibility_criteria": "Age 13 and over. Exclude pregnancy."
}' | jq


echo -e "\n======== Test 02 ========\n"

curl --silent -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "start_date": "June 2003",
  "complete_date": "November 2007",
  "icdcodes": "P70.2__O24.92__Z83.3__E10.65__E10.9__E11.65__E11.9__E10.21__E11.21__E13.21__E08.21__E09.21",
  "eligibility_criteria": "Inclusion - Type 1 or type 2 diabetes - Males and females greater than or equal to 18 years. - Abnormal kidney function determined by glomerular filtration rate - History of proteinuria - Blood pressure controlled to <140/90 on anti-hypertensive medication Exclusion - Cancer, liver disease, hepatitis, HIV+ - History of heart attack, unstable angina, stroke or peptic ulcer in the past 6 months - Pregnant or planning to become pregnant during the study period - Other known kidney disease besides diabetic nephropathy - Expect to begin dialysis or receive a kidney transplant within 1 year of study enrollment"
}' | jq
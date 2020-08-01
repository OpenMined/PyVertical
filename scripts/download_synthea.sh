#!/bin/bash
#git clone https://github.com/synthetichealth/synthea.git
cd synthea
./run_synthea -s 42 -p 5 --exporter.csv.export true
cd ..
mv synthea/output/csv/*csv data/synthea/

cd data/synthea
# Remove unnecessary files
rm allergies.csv careplans.csv devices.csv encounters.csv imaging_studies.csv
rm immunizations.csv organizations.csv payers.csv payer_transitions.csv
rm procedures.csv providers.csv supplies.csv

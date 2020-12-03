#!/bin/bash
# Download Synthea Data in the correct directory PyVertical/data/synthea
cd ../data/
GITKEEP_exists=/synthea/.gitkeep
if test -f "$GITKEEP_exists"; then
    rm /synthea/.gitkeep
fi

git clone https://github.com/synthetichealth/synthea.git

cd synthea

# Generate data
./run_synthea -s 42 -p 5000 --exporter.csv.export true
cd third-party/synthea


# Copy data to PyVertical/data
mv output/csv/*csv ../../data/

# Remove unnecessary files
cd ../../data/

rm allergies.csv careplans.csv devices.csv encounters.csv imaging_studies.csv
rm immunizations.csv organizations.csv payers.csv payer_transitions.csv
rm procedures.csv providers.csv supplies.csv

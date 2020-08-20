#!/bin/bash

# Generate data
cd third-party/synthea
./run_synthea -s 42 -p 5000 --exporter.csv.export true

# Copy data to PyVertical
mv output/csv/*csv ../../data/synthea/

cd ../../data/synthea
# Remove unnecessary files
rm allergies.csv careplans.csv devices.csv encounters.csv imaging_studies.csv
rm immunizations.csv organizations.csv payers.csv payer_transitions.csv
rm procedures.csv providers.csv supplies.csv

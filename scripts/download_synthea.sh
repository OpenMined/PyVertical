#!/bin/bash

# # Download Synthea Data 

# ----------------------------------------------
# If you choose to download Synthea on the PyVertical/data/synthea folder, execute:

# cd ../data/
# GITKEEP_exists=/synthea/.gitkeep
# if test -f "$GITKEEP_exists"; then
#     rm /synthea/.gitkeep
# fi

# git clone https://github.com/synthetichealth/synthea.git

# ----------------------------------------------


# Generate data
cd 'PATH'/synthea # Change 'PATH' to the correct path on your system
./run_synthea -s 42 -p 5000 --exporter.csv.export true


# Copy data to PyVertical/data
mv output/csv/*csv ../../data/

# Remove unnecessary files
cd ../../data/

rm allergies.csv careplans.csv devices.csv encounters.csv imaging_studies.csv
rm immunizations.csv organizations.csv payers.csv payer_transitions.csv
rm procedures.csv providers.csv supplies.csv

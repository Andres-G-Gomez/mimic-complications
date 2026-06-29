import pandas as pd
import os 

data_route = 'data/mimic'
# Load the compressed file into a DataFrame
df = pd.read_csv(os.path.join(data_route, 'hosp/patients.csv.gz'))
df2 = pd.read_csv(os.path.join(data_route, 'icu/chartevents.csv.gz'))

# Standard Python / Pandas way to count rows
total_patients = len(df)
print(f"Total Patients: {total_patients}")
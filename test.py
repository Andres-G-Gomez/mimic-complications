import pandas as pd

file_path = r"C:\Users\andre\Desktop\Git\mimic-complications\data\mimic\icu\chartevents.csv.gz"

print("Attempting to read the beginning of the file...")
try:
    # Read just a small chunk to see if the archive is completely unreadable
    df = pd.read_csv(file_path, nrows=5, compression='gzip')
    print("✅ Header and initial rows successfully read!")
    print(df.columns)
    
    print("\nAttempting to stream the first chunk...")
    for chunk in pd.read_csv(file_path, chunksize=10000, compression='gzip'):
        print(f"✅ Successfully read a chunk of shape: {chunk.shape}")
        break # Just check the first chunk
        
except Exception as e:
    print(f"❌ Failed: {e}")
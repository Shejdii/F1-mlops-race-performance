# 1. wyslij mi jak wyglada wizaulnie w srodku moj parquet. chce zobaczyc roznice z CSV moimi oczami. 2. Anty-leak to znaczy wypełnienie dziur w danych przez podstawienie czegos - np sredniej tak? 3. shift - co to dokladnie robi? w jakim sensie dziala shift? bo nie jestem pewny. 4. Czym sa rollingi? 5. w jakim sensie nasze preview jest lepsze? Co to znaczy leak przez etykiete posrednia? 6. Opisz dokladnie metode leave-one-out co ona znaczy bo jej nie znam tak na prawde. 7. Nie rozumiem tez artefaktu do konca. wiele bugow bylo wychwyconych dzieki logowaniu. 8. wyjasnij tez group-by/ 

from pathlib import Path
import pandas as pd

p = Path("artifacts/features.parquet")

print("FILE:", p)
print("SIZE (MB):", round(p.stat().st_size / 1024 / 1024, 2))

df = pd.read_parquet(p)
print("\n=== HEAD(5) ===")
print(df.head(5).to_string(index=False))

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== DESCRIBE (numeric) ===")
print(df.describe(include="number").T[["count","mean","std","min","max"]].to_string())

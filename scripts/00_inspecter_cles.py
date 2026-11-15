from pathlib import Path
from collections import Counter
import urllib.request
import io
from PIL import Image
import numpy as np

# 1. Chemin vers le fichier téléchargé (en dehors du dépôt Git)
KEYS_PATH = Path(r"C:\Users\pdelm\datasets\marine-inaturalist\train_keys.txt")

species_counts = Counter()
extensions = Counter()
first_sample = None
total_rows = 0

# 2. Lecture ligne par ligne (économise la RAM même sur des millions de lignes)
with open(KEYS_PATH, mode="r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # rsplit(",", 2) découpe depuis la droite : protège contre d'éventuelles virgules dans le nom d'espèce
        species_raw, filename, s3_uri = line.rsplit(",", 2)
        
        # Nettoyage des espaces et de l'antislash ("Abramis\ brama" -> "Abramis_brama")
        species = species_raw.strip().replace(r"\ ", "_")
        filename = filename.strip()
        s3_uri = s3_uri.strip()
        
        species_counts[species] += 1
        extensions[Path(filename).suffix.lower()] += 1
        total_rows += 1
        
        if first_sample is None:
            first_sample = (species, filename, s3_uri)

print(f"Nombre total d'images dans train_keys.txt : {total_rows}")
print(f"Nombre d'espèces (classes) : {len(species_counts)}")
print(f"Extensions recensées : {dict(extensions)}")

print("\nTop 5 des espèces les plus fréquentes :")
for sp, count in species_counts.most_common(5):
    print(f"  - {sp} : {count} images")

print("\nTop 5 des espèces les plus rares :")
for sp, count in species_counts.most_common()[-5:]:
    print(f"  - {sp} : {count} images")

# 3. Test de téléchargement sur UNE seule image (en mémoire, sans écrire sur le disque)
if first_sample:
    species, filename, s3_uri = first_sample
    https_url = s3_uri.replace(
        "s3://inaturalist-open-data/",
        "https://inaturalist-open-data.s3.amazonaws.com/"
    )
    print(f"\nTest de téléchargement pour [{species}] :")
    print(f"  URL : {https_url}")
    
    with urllib.request.urlopen(https_url, timeout=10) as response:
        image_bytes = response.read()
    
    with Image.open(io.BytesIO(image_bytes)) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb)
        print(f"  Succès ! Poids={len(image_bytes)/1024:.1f} Ko | Taille PIL (W,H)={img.size} | NumPy (H,W,C)={arr.shape}")
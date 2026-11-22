# ARCHITECTURE DU PROJET : CLASSIFICATION D'ESPÈCES MARINES (MARINE-INATURALIST)

1. **Configuration (`configs/`)** : isole tous les paramètres d'expérience (chemins, hyperparamètres, choix de l'architecture) du code source Python.
2. **Bibliothèque interne (`src/`)** : regroupe les composants réutilisables, découpés par domaine fonctionnel (`data`, `models`, `training`, `utils`).
3. **Points d'entrée (`scripts/`)** : orchestrent l'exécution complète des différentes phases du pipeline (audit des clés, téléchargement, compactage optionnel, entraînement, évaluation).

Toujours exécuter les scripts depuis la racine du dépôt avec `python -m scripts.<nom>` afin de résoudre proprement les imports `src.*` sur Windows, macOS et Linux.

---

## 0. INSTALLATION

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

# PyTorch d'abord (choisir la variante CUDA/CPU adaptée sur pytorch.org), puis le projet
pip install torch torchvision
pip install -e ".[dev]"              # dépendances du projet + pytest + ruff
```

`pip install -e .` seul installe uniquement les dépendances d'exécution ; l'extra `[dev]` ajoute les outils de test et de lint.

---

## 1. VUE D'ENSEMBLE DES GRANDS DOSSIERS

* **`configs/`** : contient les fichiers de configuration au format YAML (`default.yaml` pour l'architecture hybride `cnn_rnn`, `beefy_cnn.yaml` pour l'architecture `convolutional_resnet` de type ConvNeXt).
* **`scripts/`** : contient les scripts exécutables en ligne de commande (`python -m scripts.<nom>`). Chaque script correspond à une étape autonome du cycle de vie du projet.
* **`src/data/`** : gère l'accès au disque ou à l'archive binaire `dataset.bin`, le décodage JPEG rapide (`draft`), le redimensionnement complet (*warp* `224x224`) et l'assemblage des lots (*batchs*).
* **`src/models/`** : contient la définition des architectures de réseaux de neurones entraînées depuis zéro et un registre dynamique permettant de sélectionner un modèle depuis le YAML.
* **`src/training/`** : encapsule la mécanique d'apprentissage (boucle d'optimisation AdamW, précision mixte AMP, planificateur cosinus avec *warmup*, synchronisation DDP et calcul des métriques).
* **`tests/`** : tests `pytest` (métriques, registre de modèles, configuration, lecture disque vs `dataset.bin`, entraînement de bout en bout sur CPU avec un mini-jeu de données factice).
* **`.github/workflows/`** : intégration continue GitHub Actions (`ci.yml` : lint + tests ; `docker.yml` : vérification optionnelle de la construction de l'image).
* **`Dockerfile`** : image optionnelle (preuve de concept) pour lancer le projet dans un conteneur.
* **`src/utils/`** : regroupe les outils transverses d'infrastructure (détection matérielle CUDA/MPS/DDP/CPU, sauvegarde atomique des checkpoints, lecture de configuration et journalisation).

---

## 2. DÉTAIL DES MODULES, CLASSES ET FONCTIONS

### A. Configuration : `configs/`

* **`configs/default.yaml`** & **`configs/beefy_cnn.yaml`** :
  * `paths` : répertoires des données brutes, des manifestes (`train.csv`, `val.csv`, `class_to_idx.json`), des sauvegardes (`checkpoints`) et des journaux (`logs`).
  * `data` : nombre de classes (`2185` espèces), dimension cible des images (`image_size: 224`), cible de décodage JPEG (`decode_min_side: 224`) et graine aléatoire (`seed`).
  * `model` : nom de l'architecture enregistrée (`model.name`) et hyperparamètres structurels propres au réseau.
  * `train` : nombre d'epochs, taille de lot (`batch_size`), taux d'apprentissage (`lr`), *warmup*, régularisation (`weight_decay`), précision mixte (`use_amp`) et format mémoire (`channels_last`).

### B. Points d'entrée : `scripts/`

* **`scripts/00_inspecter_cles.py`** : audite le fichier `train_keys.txt` (nombre total d'images, répartition des espèces, extensions) et vérifie le téléchargement d'un échantillon depuis AWS S3.
* **`scripts/01_telecharger_et_couper.py`** : télécharge un échantillon stratifié en parallèle (`ThreadPoolExecutor`), convertit proprement toutes les images en JPEG RGB et génère `train.csv`, `val.csv` et `class_to_idx.json` sans fuite de données lors des ajouts incrémentaux.
* **`scripts/01b_pack_shard.py`** : compacte optionnellement les fichiers JPEG dans un fichier binaire contigu (`dataset.bin`) et met à jour les décalages (`byte_offset`, `byte_length`) dans les manifestes CSV afin d'éliminer la latence d'ouverture de millions de petits fichiers sur cluster.
* **`scripts/02_train.py`** :
  * `seed_everything(seed)` : fixe l'état des générateurs aléatoires (Python, NumPy, PyTorch CPU/CUDA) et active cuDNN benchmark ainsi que les Tensor Cores TF32 sur GPU NVIDIA RTX.
  * `CudaPrefetcher` : précharge de manière asynchrone le lot suivant en VRAM sur un flux CUDA secondaire pendant que le GPU calcule le lot courant.
  * `ExperimentRunner` : charge la configuration YAML, applique les surcharges CLI (`--data-root`, `--epochs`, `--batch-size`, `--num-workers`, `--resume`), instancie le modèle via `build_model` et lance `MarineTrainer.fit`.
* **`scripts/03_evaluate.py`** :
  * `UnlabeledImageDataset` : parcourt récursivement un répertoire d'images sans étiquettes pour l'inférence finale.
  * `ModelEvaluator.evaluate_validation()` : recharge un checkpoint, évalue le modèle sur `val.csv` (Top-1, Top-5, Macro-F1) et génère `logs/val_per_class_report.csv`.
  * `ModelEvaluator.predict_test()` : produit le fichier `submission.csv` sur le jeu de test.

### C. Données et prétraitement : `src/data/`

* **`src/data/transforms.py`** :
  * `TransformConfig` : dataclass extrayant `image_size` (`224`) depuis le YAML.
  * `MarineTransformBuilder` : construit les pipelines `build_train()` et `build_val()` autour d'un **redimensionnement complet (*warp*) `A.Resize(224, 224)`**. Ce choix évite de couper les extrémités morphologiques (nageoires, queues, tentacules, corps allongés) comme le ferait un recadrage (*crop*).
* **`src/data/dataset.py`** :
  * `SampleRecord` : enregistrement immuable et compact (`@dataclass(frozen=True, slots=True)`).
  * `MarineSpeciesDataset` : lit les manifestes CSV de manière portable entre Windows, macOS et Linux, exploite le mode `Image.draft()` de `libjpeg` et bascule automatiquement sur `dataset.bin` s'il est présent.
  * `create_dataloaders()` : instancie les `DataLoader` avec `DistributedSampler` automatique en environnement DDP multi-GPU.

### D. Architectures de réseaux : `src/models/`

* **`src/models/__init__.py`** : définit `BaseVisionClassifier`, le décorateur `@register_model`, la fabrique `build_model(cfg)` et la découverte automatique des modules `_auto_discover_models()`.
* **`src/models/cnn_rnn.py`** (`model.name: "cnn_rnn"`) : encodeur convolutif à 5 étages (`224 -> 112 -> 56 -> 28 -> 14 -> 7`) suivi d'une normalisation `LayerNorm`, d'un réseau récurrent bidirectionnel (RNN, GRU ou LSTM) parcourant les 49 jetons spatiaux et d'une moyenne globale.
* **`src/models/beefy_cnn.py`** (`model.name: "convolutional_resnet"`) : architecture purement convolutive inspirée de ConvNeXt (stem `4x4`, convolutions séparables en profondeur `7x7`, `LayerNorm2d`, `GELU` et modulation résiduelle `LayerScale`).

### E. Entraînement et utilitaires : `src/training/` & `src/utils/`

* **`src/training/trainer.py`** (`MarineTrainer`) : gère AdamW avec exclusion du `weight_decay` sur les paramètres 1D, montée linéaire (*warmup*) puis décroissance en cosinus par batch, précision mixte AMP, écrêtage de gradients (`clip_grad_norm_`) et synchronisation distribuée des métriques de validation.
* **`src/training/metrics.py`** (`ClassificationMetricTracker`) : calcule l'exactitude Top-1 et le score Macro-F1 via `torch.bincount` avec réduction globale `all_reduce` en mode DDP.
* **`src/utils/device.py`** : sélectionne automatiquement `DDP (Cluster Linux) -> CUDA (NVIDIA RTX) -> MPS (Apple Silicon) -> CPU`.
* **`src/utils/checkpoint.py`** : sauvegarde atomiquement `<modele>_last.pt` et `<modele>_best.pt` via fichiers temporaires et `os.replace()`.
* **`src/utils/logger.py`** & **`src/utils/visualize.py`** : journalisent chaque epoch dans `logs/metrics.csv` et tracent les courbes dans `logs/training_curves.png`.

---

## 3. COMMANDES D'EXÉCUTION RAPIDE (WINDOWS, MACOS, CLUSTER LINUX)

```bash
# 1. Inspecter train_keys.txt et tester l'accès S3
python -m scripts.00_inspecter_cles --data-root /chemin/vers/marine-inaturalist

# 2. Télécharger l'échantillon stratifié et créer train.csv / val.csv
python -m scripts.01_telecharger_et_couper --data-root /chemin/vers/marine-inaturalist

# 3. (Optionnel) Compacter les JPEG dans dataset.bin pour accélérer les I/O
python -m scripts.01b_pack_shard --data-root /chemin/vers/marine-inaturalist

# 4. Entraîner en mono-GPU (Windows RTX / Linux) ou Apple Silicon (macOS MPS)
python -m scripts.02_train --config configs/default.yaml --data-root /chemin/vers/marine-inaturalist

# 5. Reprendre un entraînement interrompu
python -m scripts.02_train --config configs/default.yaml --data-root /chemin/vers/marine-inaturalist --resume /chemin/vers/checkpoints/cnn_rnn_last.pt

# 6. Entraîner en distribué sur un cluster Linux multi-GPU (ex. 4 GPU NVIDIA RTX)
torchrun --nproc_per_node=4 -m scripts.02_train --config configs/beefy_cnn.yaml --data-root /scratch/marine-inaturalist

# 7. Évaluer sur validation ou générer une soumission test
python -m scripts.03_evaluate --mode val --data-root /chemin/vers/marine-inaturalist
python -m scripts.03_evaluate --mode test --data-root /chemin/vers/marine-inaturalist --test-dir /chemin/vers/test
```

---

## 4. TESTS ET INTÉGRATION CONTINUE (CI)

### Lancer les tests en local

```bash
pytest                       # tout (≈ 1 à 3 min sur CPU, dont 2 entraînements d'une epoch)
pytest -m "not slow"         # uniquement les tests rapides (sans entraînement de bout en bout)
ruff check .                 # lint (erreurs de syntaxe et pyflakes : noms indéfinis, imports inutilisés)
```

| Fichier de test | Ce qu'il vérifie |
| --- | --- |
| `tests/test_metrics.py` | Top-1 et Macro-F1 sur un cas calculé à la main, tracker vide, `reset()` |
| `tests/test_models.py` | Les deux architectures sont enregistrées, forme `(B, num_classes)` des sorties (RNN/GRU/LSTM), erreurs sur nom de modèle ou type de RNN inconnu |
| `tests/test_config.py` | Chaque YAML de `configs/` se charge et désigne un modèle existant ; `--data-root` et `MARINE_DATA_ROOT` relocalisent tous les chemins |
| `tests/test_dataset.py` | Forme des échantillons ; les lectures via `dataset.bin` donnent exactement les mêmes pixels que les lectures disque |
| `tests/test_smoke_train.py` | Une epoch complète sur CPU pour `default.yaml` et `beefy_cnn.yaml` : checkpoint `*_last.pt` et `metrics.csv` créés, pas de `crash.log` |

Le jeu de données factice (3 espèces, 18 images 64x48) est généré à la volée dans un dossier temporaire par `tests/conftest.py` : aucun téléchargement, aucune donnée réelle.

### Ce que la CI ne couvre pas

Les runners GitHub n'ont pas de GPU : la CI ne teste donc ni DDP/NCCL, ni l'AMP, ni `channels_last`/`CudaPrefetcher` sur CUDA. Ces chemins se valident à la main sur le cluster (`torchrun ...`).

### Activer la CI sur GitHub

1. Les fichiers sont déjà au bon endroit : `.github/workflows/ci.yml` à la **racine du dépôt** (le dossier `.github/workflows/` est obligatoire, GitHub ignore tout autre emplacement).
2. Faire un `git add .github tests pyproject.toml && git commit && git push` sur la branche `main` (ou ouvrir une Pull Request).
3. Ouvrir l'onglet **Actions** du dépôt : le workflow *CI* apparaît et s'exécute à chaque push sur `main` et à chaque Pull Request. Si GitHub affiche un bouton « I understand my workflows, go ahead and enable them » (fréquent sur un fork), le cliquer.
4. Lancement manuel : Actions → *CI* → **Run workflow**.
5. (Recommandé) Rendre la CI obligatoire : **Settings → Branches → Add branch protection rule** sur `main` → cocher *Require status checks to pass before merging* → sélectionner `Lint (ruff)` et `Tests (Python 3.10)` / `Tests (Python 3.12)` (ces noms n'apparaissent dans la liste qu'après une première exécution).
6. (Optionnel) Badge à coller en haut du README (remplacer `<utilisateur>/<depot>`) :
   `![CI](https://github.com/<utilisateur>/<depot>/actions/workflows/ci.yml/badge.svg)`

Le workflow installe PyTorch en version CPU (`--index-url https://download.pytorch.org/whl/cpu`) avant le projet, ce qui évite de télécharger plusieurs Go de bibliothèques CUDA inutiles.

---

## 5. DOCKER (OPTIONNEL, PREUVE DE CONCEPT)

Le projet peut être lancé dans un conteneur, mais ce n'est **pas** nécessaire : le workflow `.venv` + `pyproject.toml` reste la méthode de référence. Principes :

* l'image ne contient **que le code et les dépendances** ; données, checkpoints et logs vivent dans un volume monté sur `/data` (la variable `MARINE_DATA_ROOT=/data` relocalise tous les chemins, comme `--data-root`) ;
* les dépendances sont lues dans `pyproject.toml` (une seule liste à maintenir) ; `torch`/`torchvision` viennent de l'image de base `pytorch/pytorch`.

```bash
# Construire l'image (depuis la racine du dépôt)
docker build -t marine-baseline .

# Vérifier que ça démarre
docker run --rm marine-baseline

# Entraîner sur un GPU (nécessite le NVIDIA Container Toolkit)
docker run --rm --gpus all --ipc=host -v /chemin/vers/marine-inaturalist:/data \
  marine-baseline python -m scripts.02_train --config configs/default.yaml

# Plusieurs GPU sur une même machine
docker run --rm --gpus all --ipc=host -v /chemin/vers/marine-inaturalist:/data \
  marine-baseline torchrun --nproc_per_node=4 -m scripts.02_train --config configs/beefy_cnn.yaml

# Éviter des fichiers appartenant à root dans le volume (Linux)
docker run --rm --gpus all --ipc=host --user $(id -u):$(id -g) -v ... marine-baseline ...
```

`--ipc=host` (ou `--shm-size=8g`) est nécessaire dès que `num_workers > 0` : la mémoire partagée par défaut de Docker (64 Mo) fait planter les workers du `DataLoader`.

Une vérification optionnelle de la construction de l'image existe dans `.github/workflows/docker.yml` : elle se déclenche uniquement quand le `Dockerfile`, `.dockerignore` ou `pyproject.toml` change dans une Pull Request, ou à la demande (Actions → *Docker build* → **Run workflow**). Rien n'est publié sur un registre.

Remarque : la plupart des clusters HPC interdisent Docker. Si un jour l'image est utile là-bas, Apptainer peut la convertir (`apptainer pull marine.sif docker://...`), mais ce n'est pas couvert ici.

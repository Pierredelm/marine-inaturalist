ARCHITECTURE DU PROJET : CLASSIFICATION D'ESPECES MARINES



1. La configuration (configs/) isole tous les paramètres d'expérience (chemins, hyperparamètres, choix de l'architecture) du code source.

2. La bibliothèque interne (src/) regroupe les composants réutilisables, découpés par domaine fonctionnel (données, modèles, entraînement, utilitaires).

3. Les points d'entrée (scripts/) orchestrent l'exécution complète des différentes phases du pipeline (préparation, entraînement, évaluation).



1. VUE D'ENSEMBLE DES GRANDS DOSSIERS



configs/

  Contient les fichiers de configuration au format YAML (notamment default.yaml). C'est le point de contrôle central qui définit les hyperparamètres, les chemins d'accès aux données et l'architecture du modèle à instancier. i.e. si vous voulez modifier un paramètre dans l'execution du système (pas forcément du modèle), c'est ici.



scripts/

  Contient les scripts exécutables en ligne de commande (python -m scripts.<nom>). Chaque script correspond à une étape autonome du cycle de vie du projet : constitution du jeu de données, entraînement du modèle, puis évaluation.



src/data/

  Gère l'accès au disque, le prétraitement des images, l'augmentation de données et l'assemblage des lots (batchs) destinés au processeur graphique (GPU).



src/models/

  Contient la définition des architectures de réseaux de neurones, et un registre permettant de sélectionner dynamiquement un modèle depuis la configuration YAML.



src/training/

  Encapsule la mécanique d'apprentissage : boucle d'optimisation, calcul de la fonction de perte, planification du taux d'apprentissage et calcul des métriques.



src/utils/

  Regroupe les outils transverses d'infrastructure : gestion du matériel (CPU/GPU), sauvegarde et reprise des états d'entraînement, lecture des configurations et journalisation des résultats.





2. DETAIL DES MODULES, CLASSES ET FONCTIONS



A. Configuration : configs/



configs/default.yaml

  Regroupe l'ensemble des paramètres sous quatre sections principales :

  - paths : répertoires des données brutes, des manifestes (index CSV/JSON), des sauvegardes de modèles (checkpoints) et des journaux (logs).

  - data : nombre de classes (2 185 espèces), taille cible des images, graine aléatoire (seed) et paramètres de chargement parallèle.

  - model : nom de l'architecture enregistrée (model.name) et hyperparamètres structurels propres au réseau (dimensions, couches, dropout).

  - train : nombre d'epochs, taux d'apprentissage (lr), régularisation (weight_decay), précision mixte (use_amp) et format mémoire (channels_last).





B. Points d'entrée : scripts/



scripts/01_telecharger_et_couper.py et scripts/01b_pack_shard.py

  Assurent l'acquisition des images, la séparation stratifiée entre jeu d'entraînement (train.csv) et jeu de validation (val.csv), ainsi que le compactage des images dans un fichier binaire contigu (dataset.bin) # optionel. Ce compactage évite la latence liée à l'ouverture individuelle de millions de petits fichiers JPEG sur le système de fichiers.



scripts/02_train.py

  Point d'entrée principal pour l'entraînement d'un modèle :

  - seed_everything(seed) : fixe l'état de tous les générateurs aléatoires (Python, NumPy, PyTorch CPU et CUDA) pour garantir la reproductibilité et active les optimisations matérielles cuDNN et TF32.

  - CudaPrefetcher : encapsule le DataLoader pour transférer de manière asynchrone le batch suivant vers la mémoire vidéo (VRAM) sur un flux CUDA secondaire, pendant que le GPU exécute les calculs du batch courant.

  - ExperimentRunner : charge la configuration YAML via load_config, applique les éventuelles surcharges passées en ligne de commande (batch_size, epochs, num_workers), instancie le modèle via build_model et lance MarineTrainer.fit.

  - parse_args() : analyse les arguments du terminal (--config, --data-root, --epochs, --batch-size, --num-workers, --resume).



scripts/03_evaluate.py

  Point d'entrée dédié à l'évaluation hors ligne et à l'inférence :

  - UnlabeledImageDataset : charge un répertoire d'images sans étiquettes pour la phase de prédiction finale.

  - ModelEvaluator.evaluate_validation() : recharge un checkpoint sauvegardé, évalue le modèle sur l'ensemble de validation et calcule l'exactitude Top-1, l'exactitude Top-5 et le score Macro-F1. Génère également un rapport détaillé par espèce (val_per_class_report.csv) indiquant les confusions les plus fréquentes.

  - ModelEvaluator.predict_test() : produit les prédictions sur des images de test non étiquetées et exporte le fichier submission.csv.





C. Données et prétraitement : src/data/



src/data/transforms.py

  - TransformConfig : dataclass extrayant les paramètres de prétraitement depuis le YAML.

  - MarineTransformBuilder : construit deux chaînes de transformations distinctes via build_train() (redimensionnement, recadrage, augmentations aléatoires géométriques et colorimétriques pour limiter le surapprentissage, puis normalisation) et build_val() (redimensionnement déterministe et normalisation uniquement, afin d'évaluer le modèle dans des conditions stables).



src/data/dataset.py

  - LoaderConfig : regroupe les paramètres d'exécution des DataLoaders (taille des batchs, nombre de processus CPU parallèles).

  - MarineSpeciesDataset : lit les index CSV (train.csv ou val.csv) et extrait les images correspondantes avant de leur appliquer les transformations.

  - create_dataloaders() : instancie les DataLoaders d'entraînement et de validation en gérant le parallélisme CPU (num_workers) et l'échantillonnage distribué le cas échéant.





D. Architectures de réseaux : src/models/



src/models/__init__.py

  - BaseVisionClassifier : classe abstraite définissant l'interface obligatoire de tout modèle du projet (méthodes forward et init_weights, propriété num_trainable_params).

  - register_model(name, config_cls) : décorateur associant un nom YAML à une classe de modèle et à sa dataclass de configuration.

  - build_model(cfg) : fabrique (pattern Factory) qui instancie automatiquement l'architecture demandée par le fichier YAML ou par un checkpoint.

  - _auto_discover_models() : importe automatiquement tous les modules présents dans src/models/ au démarrage afin d'enregistrer les architectures disponibles.



src/models/cnn_rnn.py

  - ModelConfig : dataclass typée contenant les paramètres de l'architecture hybride (canaux de base, type de cellule récurrente, taille cachée, nombre de couches, dropout).

  - ConvStage : bloc élémentaire enchaînant des convolutions 2D, une normalisation par lot (BatchNorm2d), une activation ReLU et une réduction spatiale par MaxPool2d.

  - CnnRnnClassifier : architecture hybride de référence. Un encodeur convolutif à cinq étages réduit l'image (224x224) en une grille spatiale de 7x7 vecteurs de caractéristiques. Ces 49 positions sont normalisées (LayerNorm) puis parcourues séquentiellement dans les deux sens par un réseau récurrent bidirectionnel (RNN, GRU ou LSTM). La moyenne globale des états de sortie alimente enfin une couche linéaire produisant les 2 185 scores de classes (logits). L'initialisation des poids combine Kaiming (CNN), Xavier et matrices orthogonales (RNN) pour stabiliser les gradients dès la première itération.





E. Entraînement et métriques : src/training/



src/training/trainer.py

  - TrainConfig : dataclass centralisant les hyperparamètres d'optimisation (nombre d'epochs, taux d'apprentissage, durée du warmup, décroissance des poids, lissage des étiquettes, seuil de coupure des gradients).

  - BaseTrainer : prépare le modèle sur le périphérique cible (GPU, multi-GPU ou CPU) et configure le format mémoire optimal (channels_last).

  - MarineTrainer : pilote l'intégralité du processus d'apprentissage :

    - _build_param_groups() : sépare les paramètres du réseau pour appliquer la régularisation (weight decay) uniquement aux matrices de poids 2D/4D, en excluant les biais et paramètres de normalisation 1D.

    - _build_scheduler() : ajuste le taux d'apprentissage à chaque batch selon une montée linéaire initiale (warmup) suivie d'une décroissance en cosinus.

    - _step_batch() : exécute la passe avant en précision mixte (autocast), ignore le batch si la perte devient non finie (NaN/Inf) afin de préserver les moments de l'optimiseur AdamW, rétro-propage le gradient, applique un écrêtage (clip_grad_norm_) et met à jour les poids.

    - train_epoch() et validate() : exécutent respectivement un passage complet sur les données d'entraînement (avec mise à jour des poids) et sur les données de validation (sans calcul de gradient).

    - fit() : coordonne la boucle sur l'ensemble des epochs, gère la reprise après interruption (resume_path), déclenche la sauvegarde des checkpoints et enregistre les métriques dans le journal CSV.



src/training/metrics.py

  - ClassificationMetricTracker : accumule les prédictions et les pertes sur l'ensemble des batchs d'une epoch afin de calculer l'exactitude globale et le score Macro-F1 (moyenne non pondérée des scores F1 de chaque espèce, accordant le même poids aux espèces rares et fréquentes).





F. Utilitaires d'infrastructure : src/utils/



src/utils/device.py

  - setup_device(), wrap_model_for_device(), cleanup_device() : détectent automatiquement l'accélérateur matériel disponible (CUDA, MPS ou CPU), gèrent le contexte de précision mixte (GradScaler et autocast) et libèrent proprement les ressources en fin d'exécution.



src/utils/checkpoint.py

  - CheckpointManager : assure la persistance atomique de l'état d'entraînement (poids du modèle, état de l'optimiseur, état du scheduler, état du scaler AMP et états des générateurs aléatoires). Maintient deux fichiers par architecture : modele_last.pt (mis à jour à chaque epoch pour la reprise en cas d'arrêt) et modele_best.pt (mis à jour uniquement lorsque le score Macro-F1 de validation atteint un nouveau maximum).



src/utils/config.py et src/utils/logger.py

  - load_config() : lit le fichier YAML et résout les chemins de données.

  - ExperimentLogger : écrit les métriques d'entraînement et de validation à chaque fin d'epoch dans un fichier CSV structuré et journalise toute exception critique.





3. DEROULEMENT D'UNE ITERATION D'ENTRAINEMENT (FLUX DE DONNEES)



1. Initialisation : 02_train.py charge configs/default.yaml, fixe les graines aléatoires et instancie les DataLoaders ainsi que le modèle déclaré sous model.name.

2. Alimentation : Les workers CPU lisent les images, appliquent les augmentations (transforms.py) et forment des batchs que CudaPrefetcher précharge en VRAM.

3. Passe avant (Forward) : MarineTrainer transmet le batch au modèle en format channels_last et précision mixte (FP16/BF16) pour obtenir 2 185 logits par image.

4. Calcul d'erreur et rétro-propagation (Backward) : CrossEntropyLoss mesure l'écart avec les vraies étiquettes, les gradients sont calculés, écrêtés, puis appliqués par AdamW tandis que le scheduler met à jour le taux d'apprentissage.

5. Validation et sauvegarde : En fin d'epoch, le modèle est évalué sans gradient sur val.csv ; les métriques sont consignées dans logs/ et les checkpoints sont mis à jour dans checkpoints/.
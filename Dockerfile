# Preuve de concept : permet de lancer le projet dans un conteneur. Usage facultatif.
# Les données, checkpoints et logs ne sont JAMAIS dans l'image : ils vivent dans le volume monté sur /data.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Les dépendances sont lues dans pyproject.toml (source unique de vérité).
# torch / torchvision sont déjà dans l'image de base, pip les considère comme satisfaits.
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

# --data-root est facultatif : config.py lit cette variable pour relocaliser tous les dossiers
ENV MARINE_DATA_ROOT=/data \
    NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "scripts.02_train", "--help"]

package_name := "composer"

pull-models:
    docker exec -it ollama ollama pull llama3.2:1b
    docker exec -it ollama ollama pull mxbai-embed-large

up:
    docker compose -f compose.yaml up ollama -d
    @just pull-models

up-gpu:
    docker compose -f compose.yaml -f .docker-compose-configs/gpu/compose.yaml up ollama -d
    @just pull-models

down:
    docker compose down ollama

print-version:
    @echo "Current version is:" `uvx --with hatch hatch version`

[confirm]
apply-version *VERSION: print-version
    uvx --with hatch hatch version {{ VERSION }}
    sed -n "s/__version__ = '\(.*\)'/\\1/p" "{{package_name}}"/_version.py > .version.tmp
    git add "{{package_name}}"/_version.py
    uv lock --upgrade-package "{{package_name}}"
    git add uv.lock
    git commit -m "Bump version to $(cat .version.tmp)"
    if [ {{ VERSION }} != "dev" ]; then git tag -a "v$(cat .version.tmp)" -m "Bump version to $(cat .version.tmp)"; fi
    rm .version.tmp

serve-jupyter:
    uv run --with=ipython,jupyterlab,matplotlib,seaborn,h5netcdf,netcdf4,scikit-learn,scipy,xarray,"nbconvert==5.6.1" jupyter lab --notebook-dir="~"

run-ipython:
    uv run --with=ipython ipython

pull-models:
    docker exec -it ollama ollama pull llama3.2:1b
    docker exec -it ollama ollama pull mxbai-embed-large

up:
    docker compose -f compose.yaml up ollama -d
    @just pull-models

up-gpu:
    docker compose -f compose.yaml -f configs/compose/gpu/compose.yaml up ollama -d
    @just pull-models

down:
    docker compose down ollama

serve-jupyter:
    uv run --with=ipython,jupyterlab,matplotlib,seaborn,h5netcdf,netcdf4,scikit-learn,scipy,xarray,"nbconvert==5.6.1" jupyter lab --notebook-dir="~"

run-ipython:
    uv run ipython


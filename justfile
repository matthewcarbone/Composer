pull-models:
    docker exec -it ollama ollama pull llama3.2:1b

up:
    docker compose -f compose.yaml up ollama -d
    @just pull-models

up-gpu:
    docker compose -f compose.yaml -f configs/gpu/compose.yaml up ollama -d
    @just pull-models

down:
	docker compose down ollama


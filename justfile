up:
	docker compose up -d
	docker exec -it ollama ollama pull llama3.2:1b

down:
	docker compose down ollama


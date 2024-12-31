# `docker compose` for setting up Ollama server

Running your LLM locally is optimal, and required when dealing with sensitive or proprietary data. Here, I'll quickly explain how to use some of the utilities in this code to spin up your own Docker container running Ollama models of your choice.

First, it's probably best to get familiar with Ollama itself. [Ollama](https://ollama.com/) is basically the premier open-source large-language model (LLM) hosting site, and makes it extremely easy to use open-source LLMs.

Next, from the root directory of this repository, you can simply run `just up`, or `just up-gpu` (if you have GPUs available) to spin up a container called `ollama` in the background. Note that the `just` recipes are simply aliases for 

```bash
docker compose -f compose.yaml up ollama -d
```

By default, `just pull-models` will also be run, and will pull at least llama3.2:1b and mxbai-embed-large. To pull other models, you can simply execute a command inside the container via e.g.:

```bash
docker exec -it ollama ollama pull llama3.2:3b
```

Inspecting the `compose.yaml` file, you might also notice that a few environment variables are required. We recommend setting these in a `.env` or simply exporting them in your `.bashrc` file or whatnot. For Ollama, the two required environment variables (and sensible defaults) are:

```
OLLAMA_KEEP_ALIVE=-1h
OLLAMA_SERVER_PORT=11434
```

Now, how do we talk to this model? Keep in mind this Ollama container is just a service listening to a port, in this case `11434`. We can talk to it over http via curl!
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Tell me about the Bohr model of the atom.",
  "stream": false
}'
```
and of course, we can use this package to utilize it (details coming soon).

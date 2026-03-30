# AI-Fun

A playground for AI experiments and agent-based systems.  
This is a lightweight, modular CLI inspired by frameworks like LangChain, but designed for learning and rapid prototyping.

---

## Features

- **Ask questions** to an LLM
- **Interactive chat** with an LLM
- **Generate images** from prompts
- **Play the company logo minigame** (AI guessing game)

---

## Usage

All commands are run via `main.py`:

```sh
python main.py <command> [options]
```

### 1. Ask a Question

Ask a question to the LLM and get a response.

```sh
python main.py ask "What is the capital of France?"
```

**Options:**
- `-m, --model <model>`: Specify the LLM model (default: `gpt-4`)
- `-s, --stream`: Stream the response as it's generated
- `-ns, --no-stream`: Disable streaming
- `-t, --think`: Enable "thinking" mode (agent reasoning)
- `-nt, --no-think`: Disable "thinking" mode

**Example:**
```sh
python main.py ask "Explain quantum computing" -m gpt-4 -s -t
```

---

### 2. Interactive Chat

Start a chat session with the LLM.

```sh
python main.py chat
```

**Options:**
- `-m, --model <model>`: Specify the LLM model (default: `gpt-4`)
- `-sys, --system <prompt>`: Set a system prompt (optional)
- `-s, --stream`: Stream responses
- `-ns, --no-stream`: Disable streaming
- `-t, --think`: Enable "thinking" mode
- `-nt, --no-think`: Disable "thinking" mode

**Example:**
```sh
python main.py chat -m gpt-4 -sys "You are a helpful assistant." -s
```

---

### 3. Image Generation

Generate an image from a text prompt.

```sh
python main.py img "A futuristic city skyline at sunset"
```

**Options:**
- `-m, --model <model>`: Image model to use (default: `dalle-3` or similar)
- `-s, --steps <int>`: Number of inference steps (default: 4)
- `-np, --negative-prompt <prompt>`: Negative prompt to avoid certain features

**Example:**
```sh
python main.py img "A cat riding a skateboard" -m dalle-3 -s 8 -np "blurry"
```

---

### 4. Company Logo Minigame

Play the company logo guessing game.

```sh
python main.py company-logo
```

**Options:**
- `-cm, --chat-model <model>`: Chat model to use (default: `gpt-4`)
- `-pm, --prompt-model <model>`: Prompt enhancement model (default: `gpt-4`)
- `-im, --image-model <model>`: Image model to use (default: `dalle-3`)

**Example:**
```sh
python main.py company-logo -cm gpt-4 -pm gpt-3.5-turbo -im dalle-3
```

---

## Customization

You can easily swap out models or tweak options for each command.  
Check the `main.py` source for more details on available arguments.

---

## Requirements

- Python 3.8+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

---

## License

MIT License

---
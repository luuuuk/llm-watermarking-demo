# LLM Watermarking Demo

A web-based demonstration of LLM watermarking techniques, featuring multiple watermarking algorithms and interactive text generation with watermark detection.

<img width="2481" height="1280" alt="Screenshot_2026-10-04_16-55-12" src="https://github.com/user-attachments/assets/12a86123-fc70-434d-9267-f4178d1c4834" />


## Features

- **Multiple Watermarking Modes**: Choose from `none`, `hard`, `soft`, or `tournament` watermarking strategies
- **Interactive Generation**: Generate text token-by-token with full control over the watermarking process
- **Real-time Detection**: Detect watermarks in generated text with statistical confidence scores
- **Customizable Parameters**: Adjust gamma, delta, history size, and other watermarking parameters
- **Web Interface**: Built-in Flask web server with a responsive frontend
- **REST API**: Full API support for programmatic integration

## Installation

### Prerequisites

- Python 3.12 or higher
- A GPU with sufficient VRAM for running the Qwen2.5-0.5B model (recommended)
- Git for cloning dependencies

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/llm-watermarking-demo.git
cd llm-watermarking-demo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Alternatively, install dependencies directly
pip install flask transformers torch accelerate
```

### Using UV (Recommended)

This project uses `uv` for dependency management:

```bash
# Install uv if not already installed
pip install uv

# Sync dependencies
uv sync

# Or install in development mode
uv pip install -e .
```

## Usage

### Starting the Server

```bash
# From the project root
python -m llm_watermarking_demo.main

# Or using the package entry point
llm-watermarking
```

The server will start on `http://localhost:5000`.

### Running with Custom Settings

```bash
# Set environment variables (optional)
export FLASK_HOST=0.0.0.0
export FLASK_PORT=8080
python -m llm_watermarking_demo.main
```

## API Endpoints

### `POST /api/start`

Start a new text generation session.

**Request body:**
```json
{
  "prompt": "Why is the sky blue?",
  "mode": "hard",
  "gamma": 0.5,
  "delta": 2.0,
  "h": 4,
  "key": "your_secret_key",
  "max_new": 200,
  "red_words": "secret,confidential"
}
```

**Parameters:**
- `prompt` (string): The input prompt for text generation (default: "Why is the sky blue?")
- `mode` (string): Watermarking mode - "none", "hard", "soft", or "tournament" (default: "none")
- `gamma` (float): Green list fraction for hard/soft modes (0.05-0.95, default: 0.5)
- `delta` (float): Soft watermark strength (0.0-20.0, default: 2.0)
- `h` (int): Context history size for detection (1-8, default: 4)
- `key` (string): Secret key for watermark generation (default: "your_token")
- `max_new` (int): Maximum number of new tokens to generate (1-400, default: 200)
- `red_words` (string): Comma-separated list of words to exclude from generation

**Response:**
```json
{
  "session_id": "unique-session-id",
  "distribution": [
    {
      "id": 1234,
      "token": "Why",
      "prob": 0.95,
      "base_prob": 0.95,
      "green": true,
      "hits": 1,
      "trials": 1
    }
  ],
  "generated_text": "Why",
  "generated_tokens": [],
  "stats": {
    "n": 0,
    "greens": 0,
    "gamma": 0.5,
    "z": 0.0,
    "p": 1.0
  },
  "done": false
}
```

### `POST /api/sample`

Sample the next token automatically based on the current distribution and watermarking mode.

**Request body:**
```json
{
  "session_id": "your-session-id"
}
```

**Response:** Same as `/api/start` plus tournament information if in tournament mode.

### `POST /api/step`

Manually select a specific token to continue generation.

**Request body:**
```json
{
  "session_id": "your-session-id",
  "token_id": 1234
}
```

### `POST /api/auto`

Generate text automatically until completion (max_new tokens or EOS token).

**Request body:**
```json
{
  "session_id": "your-session-id"
}
```

### `POST /api/detect`

Run watermark detection on the current session.

**Request body:**
```json
{
  "session_id": "your-session-id",
  "key": "your_secret_key",
  "mode": "hard",
  "gamma": 0.5,
  "h": 4
}
```

**Response:**
```json
{
  "green_flags": [true, false, true, ...],
  "stats": {
    "n": 25,
    "greens": 18,
    "gamma": 0.5,
    "z": 2.5,
    "p": 0.006
  }
}
```

**Detection Stats:**
- `n`: Number of unique (context, token) pairs analyzed
- `greens`: Number of tokens that matched the green list
- `gamma`: Expected green list fraction
- `z`: Z-score for the detection test
- `p`: P-value (probability the text is unwatermarked)

## Watermarking Modes

### No Watermarking (`mode="none"`)
Standard text generation without any watermarking. Useful for baseline comparison.

### Hard Watermarking (`mode="hard"`)
Forces the model to only sample from a "green list" of allowed tokens. The green list is a context-dependent permutation of the vocabulary.

- **Strength**: High - easily detectable
- **Impact**: May affect text quality
- **Parameters**: `gamma` controls the size of the green list

### Soft Watermarking (`mode="soft"`)
Biases the probability distribution toward the green list tokens without completely restricting sampling.

- **Strength**: Medium - balances detectability and text quality
- **Impact**: Minimal impact on text quality
- **Parameters**: `delta` controls the strength of the bias, `gamma` controls green list size

### Tournament Watermarking (`mode="tournament"`)
Uses a tournament selection approach where multiple candidate tokens compete in rounds, with watermark information influencing the tournament outcomes.

- **Strength**: Medium to High
- **Impact**: Good balance between detectability and text quality
- **Parameters**: Tournament depth is fixed at 3 rounds (8 candidates)

## Frontend Interface

The web interface provides an interactive way to:

1. **Configure Generation**: Set prompt, mode, and parameters
2. **Generate Text**: Step through token-by-token generation
3. **Monitor Watermarks**: See which tokens are "green" (watermarked)
4. **View Statistics**: Real-time watermark detection statistics
5. **Detect Watermarks**: Run detection on completed text

## Configuration

### Environment Variables

- `FLASK_HOST`: Host address (default: 0.0.0.0)
- `FLASK_PORT`: Port number (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: False)

### Model Configuration

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. To use a different model:

1. Update `MODEL_NAME` in `main.py`
2. Ensure the model is downloaded or available locally

```python
MODEL_NAME = "your-model-name"
```

## Examples

### Basic Usage

```bash
# Start the server
python -m llm_watermarking_demo.main

# In another terminal, use curl to start a session
curl -X POST http://localhost:5000/api/start \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "mode": "hard", "gamma": 0.5}'
```

### Complete Generation Flow

```python
import requests
import json

# Start session
response = requests.post(
    "http://localhost:5000/api/start",
    json={"prompt": "Tell me a story", "mode": "soft", "gamma": 0.6, "delta": 3.0}
)
data = response.json()
session_id = data["session_id"]

# Generate automatically
while not data["done"]:
    response = requests.post(
        "http://localhost:5000/api/sample",
        json={"session_id": session_id}
    )
    data = response.json()
    print(data["generated_text"][-100:])  # Print last 100 chars

# Run detection
response = requests.post(
    "http://localhost:5000/api/detect",
    json={"session_id": session_id}
)
detection = response.json()
print(f"Detection p-value: {detection['stats']['p']}")
```

## Technical Details

### Watermarking Algorithm

This implementation uses the Kirchenbauer-style watermarking approach with several enhancements:

1. **Green List Generation**: Context-seeded permutation of the vocabulary using SHA-256 hashing
2. **Z-Test Detection**: Statistical test for watermark presence using z-scores
3. **Context Awareness**: History of recent tokens used for more robust detection
4. **Multiple Strategies**: Hard, soft, and tournament-based watermarking

### Performance Considerations

- The Qwen2.5-0.5B model requires approximately 1GB of VRAM
- For better performance on CPU-only systems, consider using smaller models
- Larger models may require significant VRAM and GPU acceleration

### Security Notes

- The watermarking is probabilistic and not cryptographically secure
- Secret keys should be kept confidential
- Detection reliability depends on text length and watermarking parameters

## Development

### Project Structure

```
llm-watermarking-demo/
├── src/
│   └── llm_watermarking_demo/
│       ├── main.py          # Flask application and watermarking logic
│       ├── __init__.py      # Package initialization
│       └── static/
│           └── index.html   # Web frontend
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

### Running Tests

```bash
# Basic import test
python -c "import llm_watermarking_demo; print('Import successful')"

# Test helper functions
python -c "
from llm_watermarking_demo.main import _key_seed, _clamp, _stats
seed = _key_seed('test')
print(f'Key seed: {seed}')
print(f'Clamp: {_clamp(0.3, 0.1, 0.9, 0.5)}')
print(f'Stats: {_stats(15, 20, 0.5)}')
"
```

### Adding New Watermarking Modes

1. Add the mode to the `MODES` tuple in `main.py`
2. Implement the watermarking logic in the `_prepare_next` function
3. Update the detection logic in `_classify` and `_score` functions
4. Add frontend support in `index.html`

## Troubleshooting

### Common Issues

**Model download errors:**
- Ensure you have internet access
- Check Hugging Face token if using private models
- Increase timeout for large model downloads

**CUDA errors:**
- Install appropriate CUDA drivers for your GPU
- Check PyTorch CUDA compatibility
- Use CPU-only mode with smaller models

**Memory errors:**
- Reduce `MAX_NEW_TOKENS` in `main.py`
- Use a smaller model
- Enable model quantization

**Port conflicts:**
- Change the port in `main.py` or use environment variables
- Ensure no other service is using port 5000

### Debug Mode

Enable debug mode for detailed error messages:

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

## License

This project is provided as-is for educational and demonstration purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:

- Bug fixes
- New watermarking algorithms
- Frontend improvements
- Performance optimizations
- Documentation enhancements

## Acknowledgments

- Based on the Kirchenbauer et al. watermarking approach
- Uses Qwen2.5 models from Alibaba Cloud
- Built with Flask and Transformers libraries
- Inspired by research in LLM watermarking and content provenance

## References

- Kirchenbauer, J., Gehring, J., Not, A., Krishnamurthy, A., Been, I., Belinkov, Y., ... & Wainwright, M. J. (2023). A Watermark for Large Language Models. arXiv preprint arXiv:2301.10226.
- Christ, P., Zalzala, H., Evans, R., & Fern, X. (2023). The Hidden Cost of Idiosyncratic Tokens in Watermarking. arXiv preprint arXiv:2310.12343.
- Fernandes, A., Ferra, D., & Fontes, C. (2023). ESI: Embedding-based Soft Watermarking for Integer Arithmetic Circuits. arXiv preprint arXiv:2305.18576.

## Support

For questions or issues, please open a GitHub issue or contact the project maintainers.

# Dorjzodov - Byte-Level Decoder-Only Language Model

![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A GPT-style decoder-only Transformer trained on raw Mongolian text using PyTorch with Apple Silicon MPS acceleration. Uses a UTF-8 byte-level tokenizer instead of external tokenizers.

## Features
-  Byte-level UTF-8 tokenizer (Mongolian Cyrillic/classical scripts)
-  Configurable decoder-only Transformer (~100M+ params)
-  MPS support for macOS (M1/M2/M3)
-  Checkpoint saving/resuming
-  Greedy & temperature sampling generation
-  Minimal interactive chat interface

## Requirements
- Python 3.8+
- PyTorch 2.2.0+
- macOS with Apple Silicon (M1/M2/M3)
- 16GB+ RAM recommended
- 5GB+ free disk space

## Project Structure
mongolian-gpt/
├── data/
│   ├── raw/          # Raw Mongolian .txt corpus
│   └── processed/    # Tokenized .npy files
├── checkpoints/      # Model checkpoints
├── outputs/          # Generated text/logs
├── src/
│   ├── config.py     # Hyperparameters
│   ├── tokenizer.py  # Byte-level tokenizer
│   ├── dataset.py    # Preprocessing/Dataset
│   ├── model.py      # Transformer model
│   ├── utils.py      # Utilities
│   ├── train.py      # Training loop
│   ├── generate.py   # Text generation
│   └── chat.py       # Interactive chatbot
├── requirements.txt
└── README.md

# Installation
Clone repository
bashgit clone <repo-url>
cd mongolian-gpt

# Install dependencies
bashpip install -r requirements.txt



# Quick Start
bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data
cp your_mongolian_text.txt data/raw/mongolian_corpus.txt

# 3. Train the model
python src/train.py --data_path data/raw/mongolian_corpus.txt --preprocess

# 4. Generate text
python src/generate.py --ckpt checkpoints/latest.pt --prompt "Сайн уу"

Data Preparation
Place your raw Mongolian text file at:
bashdata/raw/mongolian_corpus.txt
Requirements:

File must be UTF-8 encoded
Larger datasets (100MB+) recommended for better results
Mixed Cyrillic/classical Mongolian scripts supported


Training
Run preprocessing and training:
bashpython src/train.py --data_path data/raw/mongolian_corpus.txt --preprocess
Resume training from last checkpoint:
bashpython src/train.py --resume
Additional options:

--steps N — train for N steps (default from config)
--device mps|cpu|cuda — force specific device
--no_amp — disable mixed precision


Text Generation
bashpython src/generate.py --ckpt checkpoints/latest.pt --prompt "Сайн уу"
Optional parameters:

--max_new INT — number of tokens to generate (default: 100)
--temp FLOAT — sampling temperature (default: 0.8)
--greedy — force greedy decoding
--top_k INT — top-k sampling
--seed INT — random seed for reproducibility

Example:
bashpython src/generate.py --ckpt checkpoints/latest.pt --prompt "Монгол хэл бол" --max_new 50 --temp 0.7

Interactive Chat
bashpython src/chat.py --ckpt checkpoints/latest.pt
Type messages and press Enter. Type exit to quit.
Chat options:

--temp FLOAT — conversation temperature
--max_response INT — max tokens per response


Configuration
Edit src/config.py to adjust:
Model Architecture:

d_model — embedding dimension (default: 768)
n_layers — number of transformer layers (default: 12)
n_heads — attention heads (default: 12)
d_ff — feedforward dimension (default: 3072)

Training Parameters:

batch_size — per-device batch size (default: 4)
grad_accum_steps — gradient accumulation (default: 8)
num_steps — total training steps (default: 10000)
learning_rate — initial learning rate (default: 3e-4)

Sequence Length:

max_seq_len / seq_len — context window (default: 512)


Mongolian Preprocessing Notes
Unicode Normalization:
pythonimport unicodedata
text = unicodedata.normalize('NFC', text)
Preprocessing steps:

Normalize to NFC form
Standardize punctuation and newlines
Remove control characters except \n and \t
For mixed Cyrillic/classical scripts, optionally unify variants before training

Character Support:

Full Mongolian Cyrillic alphabet (А-Я, Ё, Ү, Ө)
Classical Mongolian script (ᠠ-ᡸ)
Latin characters for loanwords
Standard punctuation and numbers


Apple Silicon MPS Tips
Compatibility:

Ensure PyTorch >= 2.2.0 for stable MPS support
Some operations may fall back to CPU automatically

Memory Management:

Start with small batch_size (2-4) and increase until OOM
Use gradient accumulation (grad_accum_steps) for effective large batches
Keep max_seq_len ≤ 1024 for M3 Pro to avoid memory issues

Performance:

Mixed precision (use_amp=True) is experimental on MPS
Monitor GPU memory with Activity Monitor
Expect ~2-3x speedup over CPU training


Performance Expectations
Training Speed (M3 Pro):

~500-800 tokens/sec depending on sequence length
Default config: ~2-4 hours per 1000 steps

Memory Usage:

Default config (~100M params): ~8-12GB GPU memory
Larger models (300M+): ~16GB+ GPU memory recommended

Convergence:

Small datasets (10MB): ~2000-5000 steps
Medium datasets (100MB): ~10000-20000 steps
Large datasets (1GB+): ~50000+ steps


Troubleshooting
Common Issues:
Out of Memory:
bash# Reduce batch size
# Edit src/config.py: batch_size = 2
# Or increase gradient accumulation
# Edit src/config.py: grad_accum_steps = 16
Slow Training:

Increase grad_accum_steps instead of batch_size
Ensure MPS is available: python -c "import torch; print(torch.backends.mps.is_available())"

MPS Errors:
bash# Fall back to CPU
python src/train.py --device cpu
Poor Generation Quality:

Train for more steps
Increase model size in config
Add more diverse training data
Adjust generation temperature

File Encoding Issues:
bash# Convert to UTF-8
iconv -f <source-encoding> -t UTF-8 input.txt > data/raw/mongolian_corpus.txt

Example Output
Prompt: "Монгол улсын нийслэл"
Generated text:
Монгол улсын нийслэл Улаанбаатар хот юм. Энэ хотод олон төрлийн соёлын дурсгалт газрууд байдаг. Жишээлбэл, Гандантэгчинлэн хийд, Чингис хааны талбай зэрэг олон газар байна...

Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Development setup:
bash# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/

License
Apache-2.0 license
Copyright (c) 2025 Tsogbat Bat-Erdene
# Vedika - Sanskrit NLP Toolkit

Vedika is a comprehensive toolkit for Sanskrit text processing, offering deep learning-based tools for sandhi splitting and joining, text normalization, sentence splitting, syllabification, and tokenization.

## Features

- **Sandhi Processing**
  - Split compound Sanskrit words using attention-based neural networks
  - Join Sanskrit words with proper sandhi rules
  - Support for beam search to get multiple suggestions
- **Text Processing**
  - Syllabification
  - Tokenization
  - Sentence splitting
  - Text normalization

## Installation

```bash
# Install from PyPI
pip install vedika

```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- tqdm >= 4.62.0
- regex >= 2021.8.3

## Quick Start

### Sandhi Splitting

```python
from vedika import SanskritSplit

# Initialize splitter
splitter = SanskritSplit()

# Split a single word
result = splitter.split("रामायणम्")
print(result['split'])  # Output: राम + अयन + अम्

# Batch processing
words = ["रामायणम्", "गीतागोविन्दम्"]
results = splitter.split_batch(words)
for result in results:
    print(f"{result['input']} → {result['split']}")
```

### Sandhi Joining

```python
from vedika import SandhiJoiner

# Initialize joiner
joiner = SandhiJoiner()

# Join split words
result = joiner.join("राम+अस्ति")
print(result)  # Output: रामास्ति

# Batch processing
texts = ["राम+अस्ति", "गच्छ+अमि"]
results = joiner.join_batch(texts)
print(results)  # ['रामास्ति', 'गच्छामि']
```

## Advanced Usage

### Beam Search for Multiple Suggestions

```python
# Get multiple suggestions with beam search
result = splitter.split("रामायणम्", beam_size=3)
print(f"Best split: {result['split']}")
print(f"Confidence: {result['confidence']}")
print("Alternatives:")
for alt in result['alternatives']:
    print(f"- {alt['split']} (confidence: {alt['confidence']})")
```

### Model Information

```python
# Get model details
info = splitter.get_model_info()
print(f"Vocabulary size: {info['vocabulary_size']}")
print(f"Device: {info['device']}")
print(f"Configuration: {info['model_config']}")
```

## Project Structure

```
vedika/
├── __init__.py
├── normalizer.py
├── sandhi_join.py
├── sandhi_split.py
├── sentence_splitter.py
├── syllabification.py
├── tokenizer.py
└── data/
    ├── cleaned_metres.json
    ├── sandhi_joiner.pth
    └── sandhi_split.pth
```

## Model Architecture

The sandhi processing models use:
- Bidirectional LSTM encoder
- GRU decoder with attention
- Multi-head attention mechanism
- Character-level processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Tanuj Saxena
- Soumya Sharma

## Citation

If you use Vedika in your research, please cite:

```bibtex
@software{vedika2025,
  title={Vedika: A Sanskrit Text Processing Toolkit},
  author={Saxena, Tanuj and Sharma, Soumya},
  year={2025},
  url={https://github.com/tanuj437/vedika}
}
```

## Contact

- Email: tanuj.saxena.rks@gmail.com, soumyasharma1599@gmail.com

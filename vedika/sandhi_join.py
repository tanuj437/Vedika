"""
Sanskrit Sandhi Joiner
=====================

This module provides a neural network-based approach to join split Sanskrit words that have been separated by '+' characters. It includes an encoder-decoder architecture
Author: Tanuj Saxena and Soumya Sharma
Version: 1.0.0
License: MIT
"""

import torch
import torch.nn as nn
import math
import json
import os
import warnings
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Tanuj Saxena and Soumya Sharma"
__email__ = "tanuj.saxena.rks@gmail.com and soumyasharma1599@gmail.com"
__description__ = "Neural Based Sanskrit sandhi joining library"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class EncoderLSTM(nn.Module):
    """
    LSTM-based encoder for processing split Sanskrit text.
    
    This encoder processes character-level input sequences and produces
    contextual representations for the decoder to use.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize the encoder.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder."""
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out, hidden, cell


class AttentionMechanism(nn.Module):
    """
    Multi-head attention mechanism for the decoder.
    
    Implements scaled dot-product attention to help the decoder
    focus on relevant parts of the input sequence.
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            encoder_dim: Encoder hidden dimension
            decoder_dim: Decoder hidden dimension  
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Use the old naming convention to match saved models
        self.query_proj = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.key_proj = nn.Linear(encoder_dim * 2, decoder_dim, bias=False)
        self.value_proj = nn.Linear(encoder_dim * 2, decoder_dim, bias=False)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism."""
        batch_size, seq_len = encoder_outputs.size(0), encoder_outputs.size(1)
        
        # Handle different hidden state formats
        if decoder_hidden.dim() == 3:
            query_input = decoder_hidden[-1]
        else:
            query_input = decoder_hidden
            
        # Project to query, key, value
        query = self.query_proj(query_input).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.head_dim * self.num_heads)
        
        return self.out_proj(context), attention_weights.squeeze(2).mean(dim=1)


class DecoderGRU(nn.Module):
    """
    GRU-based decoder with attention for generating joined Sanskrit text.
    
    Uses attention mechanism to selectively focus on different parts
    of the input while generating the output sequence.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, encoder_dim: int, 
                 decoder_dim: int, attention: AttentionMechanism, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize the decoder.
        
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            encoder_dim: Encoder hidden dimension
            decoder_dim: Decoder hidden dimension
            attention: Attention mechanism instance
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embedding_dim + decoder_dim, decoder_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Use the old naming convention to match saved models
        self.fc_out = nn.Linear(decoder_dim * 2, vocab_size)
        self.num_layers = num_layers
        self.hidden_projection = nn.Linear(encoder_dim * 2, decoder_dim)

    def forward(self, input_token: torch.Tensor, hidden_state: torch.Tensor, 
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through decoder."""
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        
        context, attention_weights = self.attention(hidden_state, encoder_outputs, mask)
        context = context.unsqueeze(1)
        
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden_state = self.gru(gru_input, hidden_state)
        
        combined = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(combined)
        
        return prediction, hidden_state, attention_weights


class SandhiSeq2Seq(nn.Module):
    """
    Complete sequence-to-sequence model for Sanskrit sandhi joining.
    
    Combines encoder and decoder with attention mechanism to transform
    split Sanskrit text into properly joined form.
    """
    
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderGRU, device: str):
        """
        Initialize the complete model.
        
        Args:
            encoder: Encoder network
            decoder: Decoder network  
            device: Device to run model on
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_padding_mask(self, source: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create mask to ignore padding tokens."""
        batch_size, max_len = source.size()
        lengths = lengths.to(source.device)
        mask = torch.arange(max_len, device=source.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.float()

    def forward(self, source: torch.Tensor, target: torch.Tensor, 
                source_lengths: torch.Tensor, target_lengths: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Forward pass through complete model."""
        batch_size = source.size(0)
        target_length = target.size(1)
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, target_length, vocab_size).to(self.device)
        
        # Create padding mask
        mask = self.create_padding_mask(source, source_lengths)
        
        # Encode input sequence
        encoder_outputs, hidden, cell = self.encoder(source, source_lengths)
        
        # Initialize decoder hidden state
        num_layers = self.encoder.num_layers
        hidden = hidden.view(num_layers, 2, batch_size, -1)
        combined_hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        decoder_hidden = self.decoder.hidden_projection(combined_hidden)
        
        # Start decoding
        input_token = target[:, 0]
        
        for t in range(1, target_length):
            output, decoder_hidden, attention_weights = self.decoder(
                input_token, decoder_hidden, encoder_outputs, mask
            )
            outputs[:, t] = output
            
            # Decide next input (teacher forcing vs. model prediction)
            if teacher_forcing_ratio == 0.0:  # Inference mode
                input_token = output.argmax(1)
            else:
                import random
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                input_token = target[:, t] if use_teacher_forcing else output.argmax(1)

        return outputs


class SandhiJoiner:
    """
    Main Sanskrit Sandhi Joiner class.
    
    This is the primary interface for users to join Sanskrit words that have been
    split with '+' separators. The class handles model loading, tokenization,
    and provides easy-to-use methods for both single and batch processing.
    
    Example:
        >>> from sanskrit_sandhi import SandhiJoiner
        >>> joiner = SandhiJoiner('path/to/model.pth')
        >>> result = joiner.join("राम+अस्ति")
        >>> print(result)  # "रामास्ति" (or appropriate sandhi form)
    """
    
    def __init__(self, model_path: Optional[str] = None , device: Optional[str] = None):
        """
        Initialize the Sanskrit Sandhi Joiner.
        
        Args:
            model_path: Path to the trained model checkpoint file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if model_path is None:
        # Default to internal data path inside the vedika package
            model_path = os.path.join(os.path.dirname(__file__), "data", "sandhi_joiner.pth")
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model: Optional[SandhiSeq2Seq] = None
        self.char_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_char: Optional[Dict[int, str]] = None
        self.config: Dict = {}
        self.max_sequence_length = 100
        self.special_tokens: Dict[str, str] = {}
        
        self._load_model()
    
    def _detect_special_token_format(self) -> None:
        """
        Automatically detect the format of special tokens in the vocabulary.
        
        Different models may use different formats for special tokens like
        '<PAD>', '< PAD >', '[PAD]', etc. This method finds the correct format.
        """
        token_variants = {
            'PAD': ['<PAD>', '<pad>', '<Pad>', '<PAD >', '< PAD >', '[PAD]'],
            'SOS': ['< SOS >', '< SOS >', '<sos>', '<SOS >', '< SOS >', '[SOS]', '<START>'],
            'EOS': ['<EOS>', '<eos>', '<EOS >', '< EOS >', '[EOS]', '<END>'],
            'UNK': ['<UNK>', '<unk>', '<UNK >', '< UNK >', '[UNK]', '<UNKNOWN>']
        }
        
        self.special_tokens = {}
        
        # Try to find each special token type
        for token_type, variants in token_variants.items():
            for variant in variants:
                if variant in self.char_to_idx:
                    self.special_tokens[token_type] = variant
                    break
            
            # Fallback: try to find by expected index position
            if token_type not in self.special_tokens:
                expected_indices = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
                if token_type in expected_indices:
                    expected_idx = expected_indices[token_type]
                    if expected_idx < len(self.idx_to_char):
                        self.special_tokens[token_type] = self.idx_to_char[expected_idx]
        
        # Set defaults for any missing tokens
        defaults = {'PAD': '<PAD>', 'SOS': '< SOS >', 'EOS': '<EOS>', 'UNK': '<UNK>'}
        for token_type, default in defaults.items():
            if token_type not in self.special_tokens:
                self.special_tokens[token_type] = default
    
    def _get_special_token_index(self, token_type: str) -> int:
        """Get the index for a special token type."""
        token = self.special_tokens.get(token_type, f"<{token_type}>")
        return self.char_to_idx.get(token, 0)  # Default to 0 (usually PAD)
    
    def _load_model(self) -> None:
        """
        Load the trained model and initialize all components.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract vocabulary and configuration
            self.char_to_idx = checkpoint['char2idx']
            self.idx_to_char = checkpoint['idx2char']
            self.config = checkpoint.get('config', {})
            
            # Detect special token formats
            self._detect_special_token_format()
            
            # Get model hyperparameters
            vocab_size = len(self.char_to_idx)
            embedding_dim = self.config.get('emb_size', 256)
            encoder_hidden = self.config.get('enc_hidden', 512)
            decoder_hidden = self.config.get('dec_hidden', 512)
            num_layers = self.config.get('num_layers', 2)
            num_heads = self.config.get('num_heads', 8)
            dropout = self.config.get('dropout', 0.2)
            
            self.max_sequence_length = self.config.get('max_len', 100)
            
            # Build model architecture
            encoder = EncoderLSTM(vocab_size, embedding_dim, encoder_hidden, num_layers, dropout)
            attention = AttentionMechanism(encoder_hidden, decoder_hidden, num_heads, dropout)
            decoder = DecoderGRU(vocab_size, embedding_dim, encoder_hidden, decoder_hidden, attention, num_layers, dropout)
            
            # Create complete model
            self.model = SandhiSeq2Seq(encoder, decoder, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Sanskrit Sandhi Joiner loaded successfully")
            print(f"📱 Device: {self.device}")
            print(f"📚 Vocabulary size: {vocab_size:,}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def join(self, split_text: str, max_length: Optional[int] = None) -> str:
        """
        Join split Sanskrit text into proper sandhi form.
        
        This is the main method for joining Sanskrit words. Input should be
        words separated by '+' characters.
        
        Args:
            split_text: Input text with '+' separators (e.g., "राम+अस्ति")
            max_length: Maximum output length (defaults to model's max length)
            
        Returns:
            Joined Sanskrit text in proper sandhi form
            
        Example:
            >>> joiner = SandhiJoiner('model.pth')
            >>> result = joiner.join("देव+आलय")
            >>> print(result)  # "देवालय"
        """
        if not split_text or not split_text.strip():
            return ""
        
        # Clean and prepare input
        clean_input = split_text.strip()
        max_len = max_length or self.max_sequence_length
        
        try:
            with torch.no_grad():
                # Tokenize input sequence
                input_indices = []
                for char in clean_input[:max_len]:
                    idx = self.char_to_idx.get(char, self._get_special_token_index('UNK'))
                    input_indices.append(idx)
                
                # Pad sequence to max length
                while len(input_indices) < max_len:
                    input_indices.append(self._get_special_token_index('PAD'))
                
                # Convert to tensors
                source = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(self.device)
                source_length = torch.tensor([min(len(clean_input), max_len)], dtype=torch.long).to(self.device)
                
                # Create dummy target for inference
                target = torch.zeros(1, max_len, dtype=torch.long).to(self.device)
                target[0, 0] = self._get_special_token_index('SOS')
                target_length = torch.tensor([max_len], dtype=torch.long).to(self.device)
                
                # Generate output
                model_output = self.model(source, target, source_length, target_length, teacher_forcing_ratio=0.0)
                predictions = model_output.argmax(dim=-1)
                
                # Convert predictions back to text
                result_chars = []
                pad_idx = self._get_special_token_index('PAD')
                sos_idx = self._get_special_token_index('SOS')
                eos_idx = self._get_special_token_index('EOS')
                
                for token_idx in predictions[0][1:]:  # Skip SOS token
                    idx_value = token_idx.item()
                    
                    # Stop at EOS token
                    if idx_value == eos_idx:
                        break
                    
                    # Skip special tokens
                    if idx_value not in [pad_idx, sos_idx]:
                        char = self.idx_to_char.get(idx_value, '')
                        if char and char not in self.special_tokens.values():
                            result_chars.append(char)
                
                return ''.join(result_chars)
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to join '{split_text}': {e}")
            return split_text  # Return original text on error
    
    def join_batch(self, split_texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """
        Join multiple split texts efficiently in batch mode.
        
        Args:
            split_texts: List of input texts to join
            max_length: Maximum output length for each text
            
        Returns:
            List of joined texts corresponding to input order
            
        Example:
            >>> joiner = SandhiJoiner('model.pth')
            >>> inputs = ["राम+अस्ति", "गच्छ+अमि", "सुन्दर+आकाश"]
            >>> results = joiner.join_batch(inputs)
            >>> print(results)  # ['रामास्ति', 'गच्छामि', 'सुन्दराकाश']
        """
        if not split_texts:
            return []
        
        results = []
        for text in split_texts:
            try:
                result = self.join(text, max_length)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Warning: Failed to join '{text}': {e}")
                results.append(text)  # Keep original on error
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            Dictionary containing model statistics and configuration
            
        Example:
            >>> joiner = SandhiJoiner('model.pth')
            >>> info = joiner.get_model_info()
            >>> print(f"Model has {info['total_parameters']:,} parameters")
        """
        if not self.model:
            return {'error': 'No model loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'vocabulary_size': len(self.char_to_idx) if self.char_to_idx else 0,
            'max_sequence_length': self.max_sequence_length,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': self.config,
            'special_tokens': self.special_tokens,
            'version': __version__
        }
    
    def save_vocabulary(self, output_path: Union[str, Path]) -> None:
        """
        Save the model's vocabulary to a JSON file for external use.
        
        Args:
            output_path: Path where to save the vocabulary file
            
        Example:
            >>> joiner = SandhiJoiner('model.pth')
            >>> joiner.save_vocabulary('vocab.json')
        """
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocabulary_size': len(self.char_to_idx),
            'special_token_indices': {
                'PAD': self._get_special_token_index('PAD'),
                'SOS': self._get_special_token_index('SOS'),
                'EOS': self._get_special_token_index('EOS'),
                'UNK': self._get_special_token_index('UNK')
            },
            'detected_special_tokens': self.special_tokens,
            'max_sequence_length': self.max_sequence_length,
            'version': __version__
        }
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Vocabulary saved to {output_file}")
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text for sandhi joining.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if not text or not text.strip():
            return False, "Input text is empty"
        
        if '+' not in text:
            return False, "Input text should contain '+' separators between words"
        
        # Check for extremely long input
        if len(text) > self.max_sequence_length * 2:
            return False, f"Input text too long (max {self.max_sequence_length * 2} characters)"
        
        # Check for unsupported characters (too many unknowns)
        unknown_chars = [c for c in text if c not in self.char_to_idx and c != '+']
        if len(unknown_chars) > len(text) * 0.5:  # More than 50% unknown characters
            return False, f"Too many unsupported characters: {unknown_chars[:10]}..."
        
        return True, "Input is valid"


# ===== Convenience Functions =====

def load_joiner(model_path: Union[str, Path], device: Optional[str] = None) -> SandhiJoiner:
    """
    Convenience function to quickly load a Sanskrit Sandhi Joiner.
    
    Args:
        model_path: Path to the trained model file
        device: Device to use for inference
        
    Returns:
        Initialized SandhiJoiner instance
        
    Example:
        >>> from sanskrit_sandhi import load_joiner
        >>> joiner = load_joiner('my_model.pth')
        >>> result = joiner.join("राम+अस्ति")
    """
    return SandhiJoiner(model_path, device)


def quick_join(model_path: Union[str, Path], text: str, device: Optional[str] = None) -> str:
    """
    Quick one-shot function to join Sanskrit text without keeping model in memory.
    
    Args:
        model_path: Path to the trained model file
        text: Split text to join
        device: Device to use for inference
        
    Returns:
        Joined Sanskrit text
        
    Example:
        >>> from sanskrit_sandhi import quick_join
        >>> result = quick_join('model.pth', 'राम+अस्ति')
    """
    joiner = SandhiJoiner(model_path, device)
    return joiner.join(text)


def demo(model_path: Union[str, Path] = "sandhi_joiner.pth", test_cases: Optional[List[str]] = None) -> None:
    """
    Run a demonstration of the Sanskrit Sandhi Joiner with test cases.
    
    Args:
        model_path: Path to the trained model file
        test_cases: List of test strings (uses defaults if None)
        
    Example:
        >>> from sanskrit_sandhi import demo  
        >>> demo('model.pth')
    """
    if test_cases is None:
        test_cases = [
            "राम+अस्ति",
            "गच्छ+अमि", 
            "सुन्दर+आकाश",
            "विद्या+आलय",
            "देव+इच्छा",
            "ब्रह्म+अस्त्र",
            "सूर्य+उदय",
            "चन्द्र+मस्"
        ]
    
    try:
        print("🕉️  Sanskrit Sandhi Joiner Demo")
        print("=" * 50)
        
        joiner = SandhiJoiner()
        
        for i, test_input in enumerate(test_cases, 1):
            try:
                is_valid, message = joiner.validate_input(test_input)
                if not is_valid:
                    print(f"{i:2d}. ❌ '{test_input}' -> Invalid: {message}")
                    continue
                
                result = joiner.join(test_input)
                print(f"{i:2d}. ✅ '{test_input}' -> '{result}'")
                
            except Exception as e:
                print(f"{i:2d}. ❌ '{test_input}' -> Error: {e}")
        
        # Display model information
        info = joiner.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   • Parameters: {info['total_parameters']:,}")
        print(f"   • Vocabulary: {info['vocabulary_size']:,} characters")
        print(f"   • Device: {info['device']}")
        print(f"   • Version: {info['version']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")



def inspect_model(model_path: Union[str, Path]) -> Dict:
    """
    Inspect a model file and return detailed information without fully loading it.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information
        
    Example:
        >>> from sanskrit_sandhi import inspect_model
        >>> info = inspect_model('model.pth')
        >>> print(f"Vocab size: {info['vocab_size']}")
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        char_to_idx = checkpoint.get('char2idx', {})
        idx_to_char = checkpoint.get('idx2char', {})
        config = checkpoint.get('config', {})
        
        # Analyze vocabulary
        vocab_size = len(char_to_idx)
        sample_chars = list(char_to_idx.keys())[:20] if char_to_idx else []
        
        # Look for special tokens
        special_tokens_found = []
        for token in ['<PAD>', '< SOS >', '<EOS>', '<UNK>', '< PAD >', '< EOS >']:
            if token in char_to_idx:
                special_tokens_found.append(f"'{token}' -> {char_to_idx[token]}")
        
        return {
            'file_path': str(model_path),
            'vocab_size': vocab_size,
            'sample_characters': sample_chars,
            'special_tokens_found': special_tokens_found,
            'config': config,
            'has_model_state': 'model_state_dict' in checkpoint,
            'checkpoint_keys': list(checkpoint.keys())
        }
        
    except Exception as e:
        return {
            'file_path': str(model_path),
            'error': f"Failed to inspect model: {str(e)}",
            'vocab_size': 0,
            'sample_characters': [],
            'special_tokens_found': [],
            'config': {},
            'has_model_state': False,
            'checkpoint_keys': []
        }


# ===== Package Metadata =====

__all__ = [
    'SandhiJoiner',
    'load_joiner',
    'quick_join',
    'demo',
    'inspect_model',
    'EncoderLSTM',
    'DecoderGRU',
    'AttentionMechanism',
    'SandhiSeq2Seq'
]

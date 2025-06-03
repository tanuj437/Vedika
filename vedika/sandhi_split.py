"""
Sanskrit Sandhi Splitter 
===============================

A deep learning-based  for splitting Sanskrit compound words (sandhi).
"""

__version__ = "1.0.0"
__author__ = "Tanuj Saxena and Soumya Sharma"
__email__ = "tanuj.saxena.rks@gmail.com and soumyasharma1599@gmail.com"
__description__ = "Deep learning-based Sanskrit sandhi splitting library"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import os
from typing import List, Dict, Union, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SandhiEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for Sanskrit text processing.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_size (int): Size of character embeddings
        hidden_size (int): Hidden size of LSTM layers
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
    """
    
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder."""
        embedded = self.dropout(self.embedding(x))
        encoded_output, (hidden, cell) = self.lstm(embedded)
        encoded_output = self.layer_norm(encoded_output)
        return encoded_output, hidden, cell


class SandhiAttention(nn.Module):
    """
    Multi-head attention mechanism for sandhi splitting.
    
    Args:
        encoder_hidden_size (int): Hidden size of encoder
        decoder_hidden_size (int): Hidden size of decoder
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = decoder_hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Use original layer names to match trained model
        self.query_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size, bias=False)
        self.key_proj = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size, bias=False)
        self.value_proj = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size, bias=False)
        self.out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention mechanism."""
        batch_size, sequence_length = encoder_outputs.size(0), encoder_outputs.size(1)
        
        query_input = hidden[-1] if hidden.dim() == 3 else hidden
        
        query = self.query_proj(query_input).view(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key = self.key_proj(encoder_outputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value = self.value_proj(encoder_outputs).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, self.head_dim * self.num_heads
        )
        
        return self.out_proj(context), attention_weights.squeeze(2).mean(dim=1)


class SandhiDecoder(nn.Module):
    """
    GRU-based decoder with attention for sandhi splitting.
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_size (int): Size of character embeddings
        encoder_hidden_size (int): Hidden size of encoder
        decoder_hidden_size (int): Hidden size of decoder
        attention (SandhiAttention): Attention mechanism
        num_layers (int): Number of GRU layers
        dropout (float): Dropout rate
    """
    
    def __init__(self, vocab_size: int, embedding_size: int, encoder_hidden_size: int,
                 decoder_hidden_size: int, attention: SandhiAttention, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            embedding_size + decoder_hidden_size, decoder_hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Use original layer name to match trained model
        self.fc_out = nn.Linear(decoder_hidden_size * 2, vocab_size)
        self.num_layers = num_layers
        self.hidden_projection = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through decoder."""
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        
        context, attention_weights = self.attention(hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)
        
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        
        combined = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc_out(combined)
        
        return prediction, hidden, attention_weights


class SandhiSplitterModel(nn.Module):
    """
    Complete sequence-to-sequence model for sandhi splitting.
    
    Args:
        encoder (SandhiEncoder): Encoder module
        decoder (SandhiDecoder): Decoder module
        device (torch.device): Device for computation
    """
    
    def __init__(self, encoder: SandhiEncoder, decoder: SandhiDecoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_padding_mask(self, source: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention mechanism."""
        batch_size, max_length = source.size()
        lengths = lengths.to(source.device)
        mask = torch.arange(max_length, device=source.device).expand(
            batch_size, max_length
        ) < lengths.unsqueeze(1)
        return mask.float()


class SanskritSplit:
    """
    Main class for Sanskrit sandhi splitting using deep learning.
    
    This class provides an easy-to-use interface for splitting Sanskrit compound words
    into their constituent parts using a trained sequence-to-sequence model.
    
    Example:
        >>> splitter = SanskritSplit("path/to/model.pth")
        >>> result = splitter.split("रामायणम्")
        >>> print(result['split'])
        'राम + अयन + अम्'
    
    Args:
        model_path (str): Path to the trained model checkpoint
        device (str, optional): Device to use ('cuda', 'cpu', or 'auto')
    """
    
    def __init__(self, model_path: str = "data/sandhi_split.pth", device: Optional[str] = None):
        """Initialize the Sanskrit sandhi splitter."""
        if device is None or device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and vocabulary."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.char_to_idx = checkpoint['char2idx']
            self.idx_to_char = checkpoint['idx2char']
            self.config = checkpoint['config']
            
            # Build model architecture
            self.model = self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _build_model(self) -> SandhiSplitterModel:
        """Build the model architecture from config."""
        vocab_size = len(self.char_to_idx)
        
        encoder = SandhiEncoder(
            vocab_size=vocab_size,
            embedding_size=self.config['emb_size'],
            hidden_size=self.config['enc_hidden'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        attention = SandhiAttention(
            encoder_hidden_size=self.config['enc_hidden'],
            decoder_hidden_size=self.config['dec_hidden'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        )
        
        decoder = SandhiDecoder(
            vocab_size=vocab_size,
            embedding_size=self.config['emb_size'],
            encoder_hidden_size=self.config['enc_hidden'],
            decoder_hidden_size=self.config['dec_hidden'],
            attention=attention,
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        model = SandhiSplitterModel(encoder, decoder, self.device).to(self.device)
        return model

    def _preprocess_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert text to model input format."""
        max_length = self.config.get('max_len', 80)
        
        # Convert characters to indices
        char_indices = [
            self.char_to_idx.get(char, self.char_to_idx['<UNK>']) 
            for char in text[:max_length]
        ]
        text_length = len(char_indices)
        
        # Pad sequence
        char_indices += [self.char_to_idx['<PAD>']] * (max_length - len(char_indices))
        
        # Convert to tensors
        text_tensor = torch.tensor(
            char_indices[:max_length], dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        length_tensor = torch.tensor([text_length], dtype=torch.long).to(self.device)
        
        return text_tensor, length_tensor

    def _clean_output(self, predicted_text: str) -> str:
        """Clean the predicted output."""
        return predicted_text.rstrip('+').strip()

    def split(self, word: str, beam_size: int = 1, max_length: Optional[int] = None) -> Dict[str, Union[str, float, List]]:
        """
        Split a Sanskrit compound word into its constituent parts.
        
        Args:
            word (str): Sanskrit word to split
            beam_size (int, optional): Beam search size (1 for greedy decoding)
            max_length (int, optional): Maximum output length
            
        Returns:
            Dict containing:
                - 'input': Original input word
                - 'split': Predicted sandhi split
                - 'confidence': Confidence score (if beam_size > 1)
                - 'alternatives': Alternative splits (if beam_size > 1)
                
        Example:
            >>> result = splitter.split("रामायणम्")
            >>> print(result['split'])
            'राम + अयन + अम्'
        """
        if not word or not word.strip():
            raise ValueError("Input word cannot be empty")
            
        if max_length is None:
            max_length = self.config.get('max_len', 80)
            
        with torch.no_grad():
            source_tensor, source_length = self._preprocess_text(word)
            
            if beam_size == 1:
                return self._greedy_decode(source_tensor, source_length, max_length, word)
            else:
                return self._beam_search(source_tensor, source_length, max_length, beam_size, word)

    def _greedy_decode(self, source_tensor: torch.Tensor, source_length: torch.Tensor, 
                      max_length: int, original_word: str) -> Dict:
        """Perform greedy decoding."""
        # Create padding mask
        mask = self.model.create_padding_mask(source_tensor, source_length)
        
        # Encode input
        encoder_outputs, hidden, cell = self.model.encoder(source_tensor, source_length)
        
        # Prepare decoder hidden state
        num_layers = self.model.encoder.num_layers
        hidden = hidden.view(num_layers, 2, 1, -1)
        combined_hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        decoder_hidden = self.model.decoder.hidden_projection(combined_hidden)
        
        # Start decoding with SOS token
        input_token = torch.tensor([self.char_to_idx['< SOS >']], device=self.device)
        
        predicted_chars = []
        
        for _ in range(max_length - 1):
            output, decoder_hidden, attention_weights = self.model.decoder(
                input_token, decoder_hidden, encoder_outputs, mask
            )
            
            # Get most likely token
            predicted_token = output.argmax(dim=1)
            token_id = predicted_token.item()
            
            # Stop if EOS token
            if token_id == self.char_to_idx['<EOS>']:
                break
                
            # Add character to prediction
            if token_id not in [self.char_to_idx['<PAD>'], self.char_to_idx['< SOS >']]:
                char = self.idx_to_char.get(token_id, '<UNK>')
                predicted_chars.append(char)
                
            input_token = predicted_token
        
        predicted_split = ''.join(predicted_chars)
        predicted_split = self._clean_output(predicted_split)
        
        return {
            'input': original_word,
            'split': predicted_split,
            'confidence': None,
            'alternatives': []
        }

    def _beam_search(self, source_tensor: torch.Tensor, source_length: torch.Tensor,
                    max_length: int, beam_size: int, original_word: str) -> Dict:
        """Perform beam search decoding."""
        # Create padding mask
        mask = self.model.create_padding_mask(source_tensor, source_length)
        
        # Encode input
        encoder_outputs, hidden, cell = self.model.encoder(source_tensor, source_length)
        
        # Prepare decoder hidden state
        num_layers = self.model.encoder.num_layers
        hidden = hidden.view(num_layers, 2, 1, -1)
        combined_hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        decoder_hidden = self.model.decoder.hidden_projection(combined_hidden)
        
        # Initialize beam
        beams = [(0.0, [self.char_to_idx['< SOS >']], decoder_hidden)]
        completed_beams = []
        
        for step in range(max_length - 1):
            candidates = []
            
            for score, sequence, hidden_state in beams:
                if sequence[-1] == self.char_to_idx['<EOS>']:
                    completed_beams.append((score, sequence))
                    continue
                
                # Get predictions
                input_token = torch.tensor([sequence[-1]], device=self.device)
                output, new_hidden, _ = self.model.decoder(
                    input_token, hidden_state, encoder_outputs, mask
                )
                
                # Get top k predictions
                log_probs = torch.log_softmax(output, dim=1)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    new_score = score + top_k_probs[0, i].item()
                    new_sequence = sequence + [top_k_indices[0, i].item()]
                    candidates.append((new_score, new_sequence, new_hidden))
            
            # Select top candidates
            candidates.sort(reverse=True, key=lambda x: x[0])
            beams = candidates[:beam_size]
            
            if len(completed_beams) >= beam_size:
                break
        
        # Add remaining beams to completed
        for score, sequence, _ in beams:
            completed_beams.append((score, sequence))
        
        # Select best beam
        completed_beams.sort(reverse=True, key=lambda x: x[0])
        best_score, best_sequence = completed_beams[0]
        
        # Convert to text
        def sequence_to_text(sequence):
            chars = []
            for token_id in sequence[1:]:  # Skip SOS
                if token_id == self.char_to_idx['<EOS>']:
                    break
                if token_id not in [self.char_to_idx['<PAD>'], self.char_to_idx['< SOS >']]:
                    char = self.idx_to_char.get(token_id, '<UNK>')
                    chars.append(char)
            return self._clean_output(''.join(chars))
        
        predicted_split = sequence_to_text(best_sequence)
        
        # Get alternatives
        alternatives = []
        for score, sequence in completed_beams[1:4]:  # Top 3 alternatives
            alt_text = sequence_to_text(sequence)
            if alt_text and alt_text != predicted_split:
                alternatives.append({
                    'split': alt_text,
                    'confidence': score
                })
        
        return {
            'input': original_word,
            'split': predicted_split,
            'confidence': best_score,
            'alternatives': alternatives
        }

    def split_batch(self, words: List[str], beam_size: int = 1, 
                   show_progress: bool = True) -> List[Dict]:
        """
        Split multiple Sanskrit words in batch.
        
        Args:
            words (List[str]): List of Sanskrit words to split
            beam_size (int, optional): Beam search size
            show_progress (bool, optional): Show progress bar
            
        Returns:
            List[Dict]: List of results for each word
            
        Example:
            >>> words = ["रामायणम्", "गीतागोविन्दम्"]
            >>> results = splitter.split_batch(words)
            >>> for result in results:
            ...     print(f"{result['input']} → {result['split']}")
        """
        results = []
        
        if show_progress:
            pass
        
        for i, word in enumerate(words):
            if show_progress and i % 10 == 0:
                pass
                
            try:
                result = self.split(word, beam_size=beam_size)
                results.append(result)
            except Exception as e:
                results.append({
                    'input': word,
                    'split': '<ERROR>',
                    'confidence': 0.0,
                    'alternatives': [],
                    'error': str(e)
                })
        
        if show_progress:
            pass
        
        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model configuration and statistics
        """
        return {
            'vocab_size': len(self.char_to_idx),
            'config': self.config,
            'device': str(self.device),
            'model_path': self.model_path,
            'special_tokens': {
                'padding': '<PAD>',
                'unknown': '<UNK>',
                'start': '< SOS >',
                'end': '<EOS>'
            }
        }


# Convenience functions for quick usage
def split_word(word: str, model_path: str, beam_size: int = 1) -> str:
    """
    Quick function to split a single Sanskrit word.
    
    Args:
        word (str): Sanskrit word to split
        model_path (str): Path to model file
        beam_size (int, optional): Beam search size
        
    Returns:
        str: Predicted sandhi split
        
    Example:
        >>> split = split_word("रामायणम्", "model.pth")
        >>> print(split)
        'राम + अयन + अम्'
    """
    splitter = SanskritSplit(model_path)
    result = splitter.split(word, beam_size=beam_size)
    return result['split']


def split_words(words: List[str], model_path: str, beam_size: int = 1) -> List[str]:
    """
    Quick function to split multiple Sanskrit words.
    
    Args:
        words (List[str]): List of Sanskrit words
        model_path (str): Path to model file
        beam_size (int, optional): Beam search size
        
    Returns:
        List[str]: List of predicted splits
    """
    splitter = SanskritSplit(model_path)
    results = splitter.split_batch(words, beam_size=beam_size, show_progress=False)
    return [result['split'] for result in results]


# Example usage and testing
def demo():
    """Demonstration of the library usage."""
    print("🕉️  Sanskrit Sandhi Splitter Demo")
    print("=" * 40)
    
    # This would need an actual model file
    model_path = "sandhi_split.pth"
    
    try:
        # Initialize splitter
        splitter = SanskritSplit()
        
        # Single word splitting
        test_words = [
            "रामायणम्",
            "गीतागोविन्दम्", 
            "सुन्दरकाण्डम्",
            "महाभारतम्"
        ]
        
        print("\n📝 Single Word Splitting:")
        for word in test_words:
            result = splitter.split(word)
            print(f"  {result['input']} → {result['split']}")
        
        # Batch splitting
        print(f"\n📦 Batch Splitting ({len(test_words)} words):")
        batch_results = splitter.split_batch(test_words)
        for result in batch_results:
            print(f"  {result['input']} → {result['split']}")
        
        # Beam search example
        print(f"\n🔍 Beam Search (beam_size=3):")
        beam_result = splitter.split("रामायणम्", beam_size=3)
        print(f"  Best: {beam_result['split']}")
        print(f"  Confidence: {beam_result['confidence']:.4f}")
        if beam_result['alternatives']:
            print("  Alternatives:")
            for alt in beam_result['alternatives']:
                print(f"    - {alt['split']} (conf: {alt['confidence']:.4f})")
        
        # Model info
        print(f"\n ℹ️ Model Information:")
        info = splitter.get_model_info()
        print(f"  Vocabulary Size: {info['vocab_size']}")
        print(f"  Device: {info['device']}")
        print(f"  Max Length: {info['config'].get('max_len', 'N/A')}")
        
    except FileNotFoundError:
        print(f"❌ Model file '{model_path}' not found.")
        print("   Please provide a valid model file path.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    demo()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class AdaptivePositionalEncoder(nn.Module):
    """
    Адаптивний позиційний енкодер, який враховує семантичні зв'язки між словами.
    Використовує комбінацію синусоїдального кодування та навчальних параметрів.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Класичне синусоїдальне позиційне кодування
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        # Навчальні параметри для адаптації
        self.adaptive_layer = nn.Linear(d_model, d_model)
        self.position_weights = nn.Parameter(torch.ones(max_len))
        
        # Семантичне позиційне кодування
        self.semantic_position_layer = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask for attention
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        device = x.device
        
        # Базове позиційне кодування
        pos_encoding = self.pe[:seq_len, :].transpose(0, 1).to(device)  # (1, seq_len, d_model)
        pos_encoding = pos_encoding.expand(x.size(0), -1, -1)  # (batch_size, seq_len, d_model)
        
        # Адаптивне масштабування позицій
        position_weights = self.position_weights[:seq_len].unsqueeze(0).unsqueeze(-1).to(device)
        pos_encoding = pos_encoding * position_weights
        
        # Адаптивна трансформація
        pos_encoding = self.adaptive_layer(pos_encoding)
        
        # Семантичне позиційне кодування через self-attention
        semantic_pos, _ = self.semantic_position_layer(
            pos_encoding, pos_encoding, pos_encoding, 
            key_padding_mask=attention_mask
        )
        
        # Комбінування базового та семантичного кодування
        combined_encoding = pos_encoding + 0.3 * semantic_pos
        
        return self.dropout(x + combined_encoding)

class EnhancedContextualPositionalEncoder(nn.Module):
    """
    Покращений контекстуальний позиційний енкодер з кращим розумінням семантики.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Багаторівневе позиційне кодування
        self.local_position_encoder = AdaptivePositionalEncoder(d_model, max_len)
        
        # Глобальне контекстуальне кодування
        self.global_context_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Семантичне групування слів
        self.semantic_grouping = nn.MultiheadAttention(
            d_model, num_heads=num_heads//2, dropout=0.1, batch_first=True
        )
        
        # Відносне позиційне кодування
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * max_len - 1, num_heads)
        )
        
        # Ініціалізація параметрів
        self._init_parameters()
    
    def to(self, device):
        """Переміщення всіх компонентів на потрібний пристрій"""
        super().to(device)
        self.local_position_encoder = self.local_position_encoder.to(device)
        self.semantic_grouping = self.semantic_grouping.to(device)
        self.global_context_layer = self.global_context_layer.to(device)
        return self
    
    def _init_parameters(self):
        """Ініціалізація параметрів для стабільного тренування"""
        nn.init.normal_(self.relative_position_bias, std=0.02)
    
    def _get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """Обчислює відносне позиційне зміщення"""
        device = self.relative_position_bias.device
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions += self.max_len - 1
        
        bias = self.relative_position_bias[relative_positions]  # (seq_len, seq_len, num_heads)
        return bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            attention_mask: Attention mask
        Returns:
            Contextually encoded embeddings
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Локальне позиційне кодування
        x_with_pos = self.local_position_encoder(x, attention_mask)
        
        # 2. Семантичне групування
        semantic_context, semantic_weights = self.semantic_grouping(
            x_with_pos, x_with_pos, x_with_pos,
            key_padding_mask=attention_mask
        )
        
        # 3. Глобальне контекстуальне кодування з відносним позиційним зміщенням
        # Модифікуємо attention для врахування відносних позицій
        global_context = self.global_context_layer(
            x_with_pos + 0.2 * semantic_context,
            src_key_padding_mask=attention_mask
        )
        
        return global_context

class ImprovedWordOrderAwareEncoder(nn.Module):
    """
    Покращений енкодер, що враховує порядок слів та семантичні зв'язки.
    """
    
    def __init__(self, vocab_size: int, d_model: int, clip_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 8, max_len: int = 77):
        super().__init__()
        self.d_model = d_model
        self.clip_dim = clip_dim
        
        # Покращені ембединги слів з dropout
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Покращений позиційний енкодер
        self.positional_encoder = EnhancedContextualPositionalEncoder(
            d_model, max_len, num_heads
        )
        
        # Багатошаровий трансформер з покращеною архітектурою
        encoder_layers = []
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm для кращої стабільності
            )
            encoder_layers.append(layer)
        
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # Покращена проекція до CLIP розміру
        self.projection_layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, clip_dim),
            nn.LayerNorm(clip_dim)
        )
        
        # Адаптивне пулінг для кращого представлення послідовності
        self.adaptive_pooling = nn.MultiheadAttention(
            clip_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Ініціалізація параметрів
        self._init_parameters()
    
    def _init_parameters(self):
        """Покращена ініціалізація параметрів"""
        # Ініціалізація ембедингів
        nn.init.normal_(self.word_embeddings.weight, std=0.02)
        if self.word_embeddings.padding_idx is not None:
            nn.init.constant_(self.word_embeddings.weight[self.word_embeddings.padding_idx], 0)
        
        # Ініціалізація проекційних шарів
        for module in self.projection_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def to(self, device):
        """Переміщення всіх компонентів на потрібний пристрій"""
        super().to(device)
        self.word_embeddings = self.word_embeddings.to(device)
        self.positional_encoder = self.positional_encoder.to(device)
        for layer in self.transformer_encoder:
            layer.to(device)
        self.projection_layers = self.projection_layers.to(device)
        self.adaptive_pooling = self.adaptive_pooling.to(device)
        return self
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        Returns:
            Encoded representations (batch_size, clip_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Створення маски для padding токенів
        if attention_mask is None:
            attention_mask = (input_ids == 0).to(device)  # Припускаємо, що 0 - padding token
        else:
            attention_mask = (~attention_mask.bool()).to(device)  # Інвертуємо для PyTorch convention
        
        # 1. Ембединги слів
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.embedding_dropout(embeddings)
        
        # 2. Позиційне кодування
        embeddings = self.positional_encoder(embeddings, attention_mask)
        
        # 3. Трансформер енкодер з residual connections
        hidden_states = embeddings
        for layer in self.transformer_encoder:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # 4. Проекція до CLIP розміру
        projected = self.projection_layers(hidden_states)
        
        # 5. Адаптивне пулінг для отримання фінального представлення
        # Використовуємо CLS-подібний підхід з attention
        cls_token = projected.mean(dim=1, keepdim=True)  # (batch_size, 1, clip_dim)
        
        pooled_output, attention_weights = self.adaptive_pooling(
            cls_token, projected, projected,
            key_padding_mask=attention_mask
        )
        
        final_output = pooled_output.squeeze(1)  # (batch_size, clip_dim)
        
        # Нормалізація та обмеження для стабільності
        final_output = F.normalize(final_output, p=2, dim=-1)
        final_output = torch.clamp(final_output, -10, 10)
        
        # Перевірка на NaN
        if torch.isnan(final_output).any():
            print("Warning: NaN detected in WordOrderAwareEncoder output")
            final_output = torch.nan_to_num(final_output, nan=0.0)
        
        return final_output

# Зберігаємо зворотну сумісність
class ContextualPositionalEncoder(EnhancedContextualPositionalEncoder):
    """Клас для зворотної сумісності"""
    pass

class WordOrderAwareEncoder(ImprovedWordOrderAwareEncoder):
    """Клас для зворотної сумісності"""
    pass

def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask for input sequences.
    
    Args:
        input_ids: Input token IDs (batch_size, seq_len)
        pad_token_id: ID of padding token
        
    Returns:
        Attention mask (batch_size, seq_len) where 1 indicates valid tokens
    """
    mask = (input_ids != pad_token_id).long()
    # Ensure the mask is on the same device as input_ids
    return mask.to(input_ids.device)
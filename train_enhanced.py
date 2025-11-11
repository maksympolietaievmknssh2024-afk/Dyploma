#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Покращений скрипт навчання моделі на комбінованому датасеті
Оптимізований для довготривалого тренування (6-7 годин)
"""

import os
import shutil
# Зменшуємо фрагментацію CUDA-пам'яті
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,garbage_collection_threshold:0.6')
import argparse
import torch
# Увімкнемо TF32, якщо доступно, для кращої продуктивності при збереженні пам'яті
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import time
import random
import logging
from datetime import datetime
import gc

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from utils.text_processing import EnhancedTextProcessor

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CombinedDataset(torch.utils.data.Dataset):
    """Датасет для роботи з комбінованими даними"""
    
    def __init__(self, dataset_file, transform=None, image_size=512):
        self.transform = transform
        self.image_size = image_size
        self.text_processor = EnhancedTextProcessor()
        
        # Завантажуємо комбінований датасет
        logger.info(f"Завантаження датасету з {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Завантажено {len(self.data)} зразків")
        
        # Створюємо синтетичні зображення для промптів
        self.synthetic_images = {}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            prompt = item['prompt']
            
            # Створюємо синтетичне зображення для промпту
            image = self._create_synthetic_image(prompt, item.get('category', 'unknown'))
            
            if self.transform:
                image = self.transform(image)
            
            # Обробляємо текст
            try:
                enhanced_prompt = self.text_processor.enhance_prompt(prompt)
                text_features = enhanced_prompt
            except Exception as e:
                logger.warning(f"Помилка при обробці тексту для промпту '{prompt}': {e}")
                text_features = prompt  # Використовуємо оригінальний промпт
            
            return {
                'image': image,
                'text': prompt,
                'text_features': text_features,
                'category': item.get('category', 'unknown'),
                'id': item.get('id', f'item_{idx}')
            }
        except Exception as e:
            logger.error(f"Помилка при обробці елемента {idx}: {e}")
            # Повертаємо базовий елемент у випадку помилки
            try:
                # Гарантуємо, що повертається тензор, а не PIL.Image
                fallback_image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
                if self.transform:
                    fallback_image = self.transform(fallback_image)
                return {
                    'image': fallback_image,
                    'text': "default prompt",
                    'text_features': "default prompt",
                    'category': 'unknown',
                    'id': f'error_item_{idx}'
                }
            except Exception as et:
                logger.error(f"Помилка при формуванні fallback-елемента: {et}")
                # Остання спроба: просто повернути нульовий тензор потрібного розміру
                zero_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                return {
                    'image': zero_tensor,
                    'text': "default prompt",
                    'text_features': "default prompt",
                    'category': 'unknown',
                    'id': f'error_item_{idx}'
                }
    
    def _create_synthetic_image(self, prompt, category):
        """Створює синтетичне зображення на основі промпту"""
        try:
            # Створюємо базове зображення
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
            
            # Додаємо варіативність на основі промпту
            hash_value = hash(prompt) % 1000000
            np.random.seed(hash_value)
            
            # Створюємо кольорові плями
            pixels = np.array(image, dtype=np.uint8)
            for _ in range(5):
                x = np.random.randint(0, self.image_size)
                y = np.random.randint(0, self.image_size)
                size = np.random.randint(20, 100)
                color = np.random.randint(0, 256, 3, dtype=np.uint8)
                
                # Додаємо кольорову пляму
                for dx in range(-size//2, size//2):
                    for dy in range(-size//2, size//2):
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < self.image_size and 0 <= new_y < self.image_size:
                            if dx*dx + dy*dy <= (size//2)**2:
                                pixels[new_y, new_x] = color
            
            return Image.fromarray(pixels, mode='RGB')
        except Exception as e:
            logger.error(f"Помилка при створенні синтетичного зображення: {e}")
            # Повертаємо просте сіре зображення у випадку помилки
            return Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))

def create_data_transforms(image_size: int = 512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def train_enhanced(args):
    """Покращена функція навчання"""
    logger.info("=== ПОЧАТОК ПОКРАЩЕНОГО НАВЧАННЯ ===")
    logger.info(f"Параметри навчання:")
    logger.info(f"  Епохи: {args.num_epochs}")
    logger.info(f"  Розмір батчу: {args.batch_size}")
    logger.info(f"  Швидкість навчання: {args.learning_rate}")
    logger.info(f"  Датасет: {args.dataset_file}")
    
    # Налаштування пристрою
    if getattr(args, 'device_id', None) is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Вказано --device_id, але CUDA недоступна.")
        if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Некоректний device_id={args.device_id}. Доступні: 0..{torch.cuda.device_count()-1}")
        torch.cuda.set_device(args.device_id)
        device = torch.device(f'cuda:{args.device_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(args, 'require_cuda', False) and device.type != 'cuda':
        raise RuntimeError("Потрібен CUDA (--require_cuda), але GPU недоступний.")
    logger.info(f"Використовується пристрій: {device}")
    
    if torch.cuda.is_available():
        # Увімкнути вибір найкращих алгоритмів cuDNN для фіксованого розміру
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"Доступна пам'ять GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Створюємо датасет та завантажувач
    transform = create_data_transforms(args.image_size)
    dataset = CombinedDataset(args.dataset_file, transform=transform, image_size=args.image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        # Оптимізації завантаження даних
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )
    
    logger.info(f"Створено датасет з {len(dataset)} зразків")
    logger.info(f"Кількість батчів на епоху: {len(dataloader)}")
    
    # Ініціалізуємо модель
    model = ImageGenerationModel(device=device)
    model = model.to(device)
    # За потреби переводимо UNet у BF16, щоб знизити пікове використання VRAM на Windows
    try:
        if device.type == 'cuda' and getattr(args, 'amp_dtype', 'fp32') == 'bf16':
            model.unet.to(torch.bfloat16)
            logger.info("UNet переведено у BF16 для економії VRAM")
    except Exception as e:
        logger.warning(f"Не вдалося перевести UNet у BF16: {e}")
    
    # Відновлення з попередньо збережених ваг (теплий старт)
    if getattr(args, 'resume_from', None):
        if os.path.exists(args.resume_from):
            try:
                model.load_model(args.resume_from)
                logger.info(f"Завантажено ваги моделі з: {args.resume_from}")
            except Exception as e:
                logger.warning(f"Не вдалося завантажити ваги з {args.resume_from}: {e}")
        else:
            logger.warning(f"Шлях для відновлення не існує: {args.resume_from}")
    
    # Налаштовуємо оптимізатор з покращеними параметрами
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # AMP: налаштування типу та скейлера
    amp_enabled = (device.type == 'cuda') and (args.amp_dtype != 'fp32')
    amp_dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    autocast_dtype = amp_dtype_map.get(args.amp_dtype, torch.float32)
    # Скейлер потрібен лише для FP16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp_dtype == 'fp16'))
    
    # Налаштовуємо планувальник швидкості навчання
    T_0 = max(1, args.num_epochs // 4)  # Мінімум 1 епоха для першого циклу
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,  # Перший цикл
        T_mult=2,  # Множник для наступних циклів
        eta_min=args.learning_rate * 0.01  # Мінімальна швидкість навчання
    )
    
    # Метрики для відстеження
    training_losses = []
    semantic_losses = []
    best_loss = float('inf')

    # ===== Авто‑відновлення прогресу, якщо доступні останні файли стану =====
    start_epoch = 0
    latest_ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
    latest_state_path = os.path.join(args.output_dir, "latest_training_state.pt")
    if getattr(args, 'resume_from', None) is None:
        try:
            if os.path.exists(latest_ckpt_path):
                model.load_model(latest_ckpt_path)
                logger.info(f"Авто‑відновлення: завантажено ваги з {latest_ckpt_path}")
            if os.path.exists(latest_state_path):
                state = torch.load(latest_state_path, map_location='cpu')
                if 'optimizer_state_dict' in state:
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                try:
                    if 'scheduler_state_dict' in state:
                        scheduler.load_state_dict(state['scheduler_state_dict'])
                except Exception as e:
                    logger.warning(f"Не вдалося відновити стан планувальника: {e}")
                start_epoch = int(state.get('epoch_completed', 0))
                try:
                    best_loss = float(state.get('best_loss', best_loss))
                except Exception:
                    pass
                logger.info(f"Авто‑відновлення: продовжуємо з епохи {start_epoch+1}")
        except Exception as e:
            logger.warning(f"Авто‑відновлення: не вдалося відновити стан: {e}")
    
    # Початок навчання
    start_time = time.time()
    logger.info("Початок навчання...")
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_semantic_loss = 0.0
        batch_count = 0
        
        # Прогрес-бар для епохи
        progress_bar = tqdm(
            dataloader,
            desc=f"Епоха {epoch+1}/{args.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Якщо задано обмеження на кількість батчів для швидкого smoke‑тесту — зупиняємось
            if getattr(args, 'max_batches', None) is not None and (batch_idx + 1) > int(args.max_batches):
                break
            # zero_grad лише на початку вікна акумуляції
            if batch_idx % max(1, args.grad_accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)
            
            try:
                # Переносимо дані на пристрій
                images = batch['image'].to(device, non_blocking=True)
                texts = batch['text']
                
                # Обчислення з автокастом (AMP) з керуванням dtype
                with torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp_enabled):
                    # Використовуємо чисті зображення: модель сама кодує у латенти і додає шум
                    loss = model(images, texts)
                    
                    # Основна втрата вже обчислена в моделі
                    mse_loss = loss
                    
                    # Семантична втрата з додатковим захистом від NaN/Inf
                    semantic_loss = 0.0
                    if hasattr(model, 'semantic_contextual_encoder'):
                        try:
                            # Обчислюємо семантичні ембедінги БЕЗ градієнтів для економії VRAM
                            semantic_embeddings = []
                            for text in texts:
                                with torch.no_grad():
                                    embedding = model.semantic_contextual_encoder.encode_text(text)
                                semantic_embeddings.append(embedding)
                            
                            if semantic_embeddings:
                                # Обчислюємо семантичну регуляризацію повністю на CPU — не переносимо великі тензори на GPU
                                semantic_tensor_cpu = torch.stack(semantic_embeddings).float()
                                norms_cpu = torch.norm(semantic_tensor_cpu, dim=-1)
                                norms_cpu = torch.nan_to_num(norms_cpu, nan=0.0, posinf=1.0, neginf=0.0)
                                # Отримуємо скаляр (CPU) та лише його додаємо до втрати на GPU
                                semantic_loss = float(torch.mean(norms_cpu).item() * 0.01)
                        except Exception as e:
                            logger.warning(f"Помилка в семантичній втраті: {e}")
                            semantic_loss = 0.0
                    
                    # Загальна втрата (семантика не впливає на градієнт Unet)
                    total_loss = mse_loss + (semantic_loss if isinstance(semantic_loss, torch.Tensor) else torch.tensor(semantic_loss, device=device))
                
                # Перевіряємо на NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning(f"Виявлено NaN/Inf втрату в батчі {batch_idx}, пропускаємо")
                    continue
                
                # Зворотний прохід з акумуляцією: з урахуванням скейлера лише для FP16
                if scaler.is_enabled():
                    scaler.scale(total_loss / max(1, args.grad_accum_steps)).backward()
                else:
                    (total_loss / max(1, args.grad_accum_steps)).backward()
                
                # Обрізання градієнтів та крок оптимізатора лише коли завершили акумуляцію
                if (batch_idx + 1) % max(1, args.grad_accum_steps) == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Оновлюємо метрики
                total_loss_value = total_loss.item()
                semantic_loss_value = semantic_loss if isinstance(semantic_loss, (int, float)) else semantic_loss.item()
                
                epoch_loss += total_loss_value
                epoch_semantic_loss += semantic_loss_value
                batch_count += 1
                
                # Оновлюємо прогрес-бар
                progress_bar.set_postfix({
                    'Loss': f'{total_loss_value:.4f}',
                    'Sem': f'{semantic_loss_value:.4f}',
                    'Avg': f'{epoch_loss/batch_count:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                # Розширена діагностика VRAM
                try:
                    if torch.cuda.is_available() and (batch_idx < 20 or (batch_idx + 1) % max(1, args.log_mem_every) == 0):
                        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                        mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                        logger.info(f"GPU mem batch {batch_idx+1}: alloc={mem_alloc:.1f}MB, reserved={mem_reserved:.1f}MB")
                        # авто-очистка кешу за потреби
                        if mem_reserved - mem_alloc > 512 and (batch_idx + 1) % max(1, args.log_mem_every) == 0:
                            torch.cuda.empty_cache()
                except Exception:
                    pass
                # Явний лог кожні 50 батчів для видимого прогресу в терміналі
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Епоха {epoch+1}, батч {batch_idx+1}/{len(dataloader)}: loss={total_loss_value:.4f}, sem={semantic_loss_value:.4f}, avg={epoch_loss/batch_count:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                    # Додатково чистимо кеш GPU для стабільності при довгих ранках
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"OOM в батчі {batch_idx}: {e}. Очищаю кеш і пропускаю батч.")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                gc.collect()
                continue
            except Exception as e:
                logger.error(f"Помилка в батчі {batch_idx}: {e}")
                continue
        
        # Оновлюємо планувальник
        scheduler.step()
        
        # Статистика епохи
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            avg_semantic_loss = epoch_semantic_loss / batch_count
            training_losses.append(avg_epoch_loss)
            semantic_losses.append(avg_semantic_loss)
            
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (epoch + 1)) * args.num_epochs
            remaining_time = estimated_total_time - elapsed_time
            
            logger.info(f"Епоха {epoch+1} завершена:")
            logger.info(f"  Середня втрата: {avg_epoch_loss:.4f}")
            logger.info(f"  Семантична втрата: {avg_semantic_loss:.4f}")
            logger.info(f"  Поточна швидкість навчання: {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"  Час виконання: {elapsed_time/60:.1f} хв")
            logger.info(f"  Залишилось часу: {remaining_time/60:.1f} хв")
            
            # Зберігаємо найкращу модель
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                try:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                    model.save_model(best_model_path)
                    logger.info(f"  Збережено найкращу модель: {best_model_path}")
                except Exception as e:
                    logger.warning(f"Не вдалося зберегти найкращу модель {best_model_path}: {e}")
        
        # Зберігаємо чекпоінт за розкладом
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            # На Windows інколи виникають проблеми із zip‑серіалізацією/IPC під час сейву.
            # Перед збереженням прибираємо зайві CUDA ресурси, а сам сейв обгортаємо у try/except
            try:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                model.save_model(checkpoint_path)
                logger.info(f"  Збережено чекпоінт: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Не вдалося зберегти чекпоінт {checkpoint_path}: {e}")

        # Завжди оновлюємо latest_checkpoint для авто‑резюму
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            model.save_model(latest_ckpt_path)
        except Exception as e:
            logger.warning(f"Не вдалося зберегти latest_checkpoint: {e}")

        # Зберігаємо метрики атомарно після кожної епохи
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        try:
            tmp_metrics_path = metrics_path + ".tmp"
            with open(tmp_metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'losses': training_losses,
                    'semantic_losses': semantic_losses,
                    'epochs_completed': epoch + 1,
                    'total_epochs': args.num_epochs,
                    'best_loss': best_loss,
                    'current_lr': scheduler.get_last_lr()[0],
                    'training_time_minutes': (time.time() - start_time) / 60,
                    'estimated_remaining_minutes': remaining_time / 60
                }, f, indent=2, ensure_ascii=False)
            os.replace(tmp_metrics_path, metrics_path)
        except Exception as e:
            logger.warning(f"Не вдалося зберегти метрики: {e}")

        # Зберігаємо останній стан тренування (epoch, optimizer, scheduler)
        try:
            latest_state_tmp = os.path.join(args.output_dir, "latest_training_state.pt.tmp")
            latest_state_path = os.path.join(args.output_dir, "latest_training_state.pt")
            # Конвертуємо стани оптимізатора на CPU, щоб уникнути CUDA IPC/zip‑серіалізації на Windows
            opt_state = optimizer.state_dict()
            try:
                for s in opt_state.get('state', {}).values():
                    keys = list(s.keys())
                    for k in keys:
                        v = s[k]
                        if isinstance(v, torch.Tensor):
                            s[k] = v.detach().cpu()
            except Exception as e:
                logger.warning(f"Не вдалося повністю перенести стани оптимізатора на CPU: {e}")
            # Використовуємо weights_only якщо доступний, інакше legacy без zip
            state_obj = {
                'epoch_completed': epoch,
                'best_loss': best_loss,
                'optimizer_state_dict': opt_state,
                'scheduler_state_dict': scheduler.state_dict(),
            }
            try:
                torch.save(state_obj, latest_state_tmp, weights_only=True)
            except TypeError:
                torch.save(state_obj, latest_state_tmp, _use_new_zipfile_serialization=False)
            try:
                os.replace(latest_state_tmp, latest_state_path)
            except Exception:
                shutil.copyfile(latest_state_tmp, latest_state_path)
                try:
                    os.remove(latest_state_tmp)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Не вдалося зберегти latest_training_state: {e}")
    
    # Зберігаємо фінальну модель
    final_model_path = os.path.join(args.output_dir, "final_enhanced_model.pt")
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        model.save_model(final_model_path)
    except Exception as e:
        logger.warning(f"Не вдалося зберегти фінальну модель: {e}")
    
    total_time = time.time() - start_time
    logger.info("=== НАВЧАННЯ ЗАВЕРШЕНО ===")
    logger.info(f"Фінальна модель: {final_model_path}")
    logger.info(f"Найкраща модель: {os.path.join(args.output_dir, 'best_model.pt')}")
    logger.info(f"Загальний час навчання: {total_time/3600:.2f} годин")
    logger.info(f"Найкраща втрата: {best_loss:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Покращене навчання моделі генерації зображень")
    parser.add_argument("--dataset_file", type=str, default="combined_dataset_train.json", 
                       help="Файл комбінованого датасету")
    parser.add_argument("--output_dir", type=str, default="./enhanced_output", 
                       help="Директорія для збереження моделі")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Розмір батчу (зменшено для стабільності)")
    parser.add_argument("--num_epochs", type=int, default=200, 
                       help="Кількість епох (збільшено для кращого навчання)")
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                       help="Швидкість навчання (зменшено для стабільності)")
    parser.add_argument("--save_every", type=int, default=20, 
                       help="Зберігати чекпоінт кожні N епох")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Кількість воркерів для завантаження даних")
    parser.add_argument("--prefetch_factor", type=int, default=4, 
                       help="Кількість батчів, що префетчаться у воркерів")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Шлях до збережених ваг для теплого старту")
    parser.add_argument("--image_size", type=int, default=384,
                       help="Розмір вхідного зображення (пікселі, квадрат)" )
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Кількість кроків акумуляції градієнтів перед оновленням")
    parser.add_argument("--log_mem_every", type=int, default=50,
                       help="Як часто логувати VRAM (у батчах)")
    parser.add_argument("--device_id", type=int, default=None,
                       help="Індекс CUDA-пристрою (наприклад, 0). Якщо не задано — авто.")
    parser.add_argument("--require_cuda", action="store_true",
                       help="Помилка, якщо CUDA недоступна (без fallback на CPU).")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                       help="Тип для AMP автокасту: fp16, bf16 або fp32 (вимикає AMP).")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Максимальна кількість батчів на епоху (для швидкого smoke‑тесту)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Створюємо директорію для виводу
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Запускаємо навчання
    train_enhanced(args)
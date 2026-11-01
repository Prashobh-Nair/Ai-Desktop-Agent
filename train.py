"""
Training script for Custom Speech-to-Text Model (CNN + BiLSTM + CTC).
"""
import os
import time
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from src.text import Tokenizer, CTCGreedyDecoder
from src.model import SpeechRecognitionModel
from src.dataset import SpeechDataset, speech_collate_fn
from src.metrics import compute_wer, compute_cer


class BucketBatchSampler(Sampler):
    """
    Groups samples of similar duration into batches and shuffles the batches.
    Significantly minimizes padding computation for speech training.
    """
    def __init__(self, dataset, batch_size=8, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = [
            list(range(i, min(i + batch_size, len(dataset))))
            for i in range(0, len(dataset), batch_size)
        ]

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return len(self.batches)


def train_one_epoch(model, dataloader, criterion, optimizer, decoder, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        waveforms = batch["waveforms"].to(device)
        waveform_lens = batch["waveform_lens"].to(device)
        targets = batch["targets"].to(device)
        target_lens = batch["target_lens"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        # logits: [batch, time, vocab_size]
        logits, output_lens = model(waveforms, input_lengths=waveform_lens)
        
        # CTC Loss requires:
        # log_probs: [time, batch, vocab_size]
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
        
        loss = criterion(log_probs, targets, output_lens, target_lens)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[!] Warning: NaN/Inf loss encountered, skipping step.")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)


def validate(model, dataloader, criterion, decoder, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_refs = []
    all_hyps = []
    
    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            waveform_lens = batch["waveform_lens"].to(device)
            targets = batch["targets"].to(device)
            target_lens = batch["target_lens"].to(device)
            ref_texts = batch["texts"]
            
            logits, output_lens = model(waveforms, input_lengths=waveform_lens)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            
            loss = criterion(log_probs, targets, output_lens, target_lens)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
                
            # Decode predictions
            hypotheses = decoder.decode(logits, output_lens)
            all_refs.extend(ref_texts)
            all_hyps.extend(hypotheses)
            
    val_loss = total_loss / max(1, num_batches)
    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)
    
    sample_pairs = list(zip(all_refs[:3], all_hyps[:3]))
    return val_loss, wer, cer, sample_pairs


def main():
    parser = argparse.ArgumentParser(description="Train Custom STT Model")
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/val.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Initialize Tokenizer and Decoder
    tokenizer = Tokenizer()
    decoder = CTCGreedyDecoder(tokenizer)

    # Initialize Datasets and Loaders
    train_dataset = SpeechDataset(args.train_csv, tokenizer)
    val_dataset = SpeechDataset(args.val_csv, tokenizer)
    
    train_sampler = BucketBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_sampler = BucketBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=speech_collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=speech_collate_fn,
        num_workers=0
    )

    print(f"[*] Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Initialize Model
    model = SpeechRecognitionModel(
        n_mels=80,
        vocab_size=tokenizer.vocab_size,
        cnn_channels=[32, 64],
        rnn_hidden=192,
        rnn_layers=2,
        rnn_dropout=0.2
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total trainable parameters: {total_params:,}")

    # CTC Loss (blank token index is 0)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2
    )

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    last_ckpt_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"[*] Resuming training from {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[*] Resumed at epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

    print("\n" + "="*70)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val CER':<9} | {'Val WER':<9} | {'Time (s)':<8}")
    print("="*70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, decoder, device)
        val_loss, val_wer, val_cer, sample_pairs = validate(model, val_loader, criterion, decoder, device)
        scheduler.step(val_loss)
        
        elapsed = time.time() - t0
        
        print(f"{epoch:<6} | {train_loss:<12.4f} | {val_loss:<10.4f} | {val_cer:<9.4f} | {val_wer:<9.4f} | {elapsed:<8.1f}")
        
        # Periodically show sample prediction
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            if sample_pairs:
                print(f"    [Sample Target]:     {sample_pairs[0][0]}")
                print(f"    [Sample Prediction]: {sample_pairs[0][1]}")

        # Checkpoint saving
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_cer": val_cer,
            "best_val_loss": best_val_loss,
            "config": {
                "n_mels": 80,
                "vocab_size": tokenizer.vocab_size,
                "cnn_channels": [32, 64],
                "rnn_hidden": 192,
                "rnn_layers": 2
            }
        }
        torch.save(ckpt_data, last_ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, best_ckpt_path)
            print(f"    --> Saved new best model checkpoint to {best_ckpt_path} (val_loss: {val_loss:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "lr": optimizer.param_groups[0]["lr"]
        })

    # Save training logs to json
    log_path = os.path.join(args.checkpoint_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training completed. Logs saved to {log_path}")


if __name__ == "__main__":
    main()


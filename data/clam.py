import os
import torch
import torchaudio
import pretty_midi
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

###########################################
# Melody Encoder (Example)
###########################################


class MelodyEncoder(nn.Module):
    def __init__(self, num_pitch=128, num_duration=512, pitch_emb_dim=64, dur_emb_dim=64, hidden_dim=256, proj_dim=32):
        super(MelodyEncoder, self).__init__()
        self.pitch_emb = nn.Embedding(num_pitch, pitch_emb_dim)
        self.duration_emb = nn.Embedding(num_duration, dur_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pitch_emb_dim + dur_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, pitch_tokens, duration_tokens):
        # pitch_tokens: (B, N)
        # duration_tokens: (B, N)
        pitch_vecs = self.pitch_emb(pitch_tokens)       # (B, N, pitch_emb_dim)
        dur_vecs = self.duration_emb(duration_tokens)   # (B, N, dur_emb_dim)
        note_repr = torch.cat([pitch_vecs, dur_vecs], dim=-1) # (B, N, pitch_emb_dim+dur_emb_dim)

        # Average pooling across notes
        melody_repr = note_repr.mean(dim=1)  # (B, pitch_emb_dim+dur_emb_dim)
        melody_embs = self.mlp(melody_repr)  # (B, proj_dim)
        return melody_embs


###########################################
# CLMP Model
###########################################


class CLMPModel(nn.Module):
    def __init__(self, clap_model, melody_encoder, final_dim=512):
        super(CLMPModel, self).__init__()
        self.audio_model = clap_model.audio_model
        self.text_model = clap_model.text_model
        # clap_model.audio_model.config.hidden_size and clap_model.text_model.config.hidden_size
        # can be inspected for dimensions.
        audio_hidden = self.audio_model.config.hidden_size
        text_hidden = self.text_model.config.hidden_size
        self.melody_encoder = melody_encoder

        # Projection layers to shared embedding space
        self.audio_proj = nn.Linear(audio_hidden, final_dim)
        self.text_proj = nn.Linear(text_hidden, final_dim)
        # melody_encoder outputs a proj_dim (e.g. 768), adjust if different
        self.melody_proj = nn.Linear(32, final_dim)

    def forward(self, audio_features, input_ids, attention_mask, pitch_tokens, duration_tokens):
        # audio_features: dict returned by processor.feature_extractor with 'input_features'
        # text_inputs: input_ids, attention_mask
        # Melody tokens: pitch_tokens, duration_tokens

        # Audio forward
        audio_outputs = self.audio_model(**audio_features)
        # If audio_outputs.last_hidden_state is (B, seq_len, hidden_size),
        # just average over seq_len to get (B, hidden_size)
        print("last_hidden_state shape:", audio_outputs.last_hidden_state.shape)
        audio_emb = audio_outputs.last_hidden_state.mean(dim=(2,3)) # (B, hidden_size)
        audio_emb = self.audio_proj(audio_emb)                   # (B, final_dim)


        # Text forward
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # text_outputs.last_hidden_state: (B, seq_len, hidden_size)
        text_emb = text_outputs.last_hidden_state.mean(dim=1)
        text_emb = self.text_proj(text_emb)

        # Melody forward
        melody_emb = self.melody_encoder(pitch_tokens, duration_tokens)
        melody_emb = self.melody_proj(melody_emb)

        return audio_emb, text_emb, melody_emb


###########################################
# Symmetric Triple-Contrastive Loss
###########################################


class SymmetricTripleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SymmetricTripleContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_embs, text_embs, melody_embs):
        # Normalize
        a = F.normalize(audio_embs, p=2, dim=1)
        t = F.normalize(text_embs, p=2, dim=1)
        m = F.normalize(melody_embs, p=2, dim=1)

        batch_size = a.size(0)
        labels = torch.arange(batch_size, device=a.device)

        # Compute similarity
        sim_a_t = (a @ t.T) / self.temperature
        sim_t_a = sim_a_t.T

        sim_a_m = (a @ m.T) / self.temperature
        sim_m_a = sim_a_m.T

        sim_t_m = (t @ m.T) / self.temperature
        sim_m_t = sim_t_m.T

        # Compute losses
        loss_a_t = F.cross_entropy(sim_a_t, labels)
        loss_t_a = F.cross_entropy(sim_t_a, labels)
        loss_a_m = F.cross_entropy(sim_a_m, labels)
        loss_m_a = F.cross_entropy(sim_m_a, labels)
        loss_t_m = F.cross_entropy(sim_t_m, labels)
        loss_m_t = F.cross_entropy(sim_m_t, labels)

        loss = (loss_a_t + loss_t_a + loss_a_m + loss_m_a + loss_t_m + loss_m_t) / 6.0
        return loss


###########################################
# Example Dataset (Skeleton)
###########################################


class AudioTextMelodyDataset(Dataset):
    def __init__(self, audio_dir, melody_dir, captions_file, processor, tokenizer, max_length=128):
        self.audio_dir = audio_dir
        self.melody_dir = melody_dir

        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [l.strip() for l in f if l.strip()]

        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.melody_files = sorted([f for f in os.listdir(melody_dir) if f.endswith('.mid')])

        assert len(self.captions) == len(self.audio_files) == len(self.melody_files), "Mismatch in data lengths."

        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # Load audio
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sr = torchaudio.load(audio_path)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
            waveform = resampler(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

        # Process audio with CLAP processor
        audio_features = self.processor.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=48000,
            return_tensors="pt",
        )
        # Squeeze out the batch dimension from the feature extractor output
        audio_features = {k: v.squeeze(0) for k, v in audio_features.items()}

        # Load and process melody from .mid file
        melody_path = os.path.join(self.melody_dir, self.melody_files[idx])
        pitch_tokens, duration_tokens = self.extract_melody_tokens(melody_path)

        # Tokenize text
        caption = self.captions[idx]
        encoded = self.tokenizer(
            caption, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'audio_features': audio_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pitch_tokens': pitch_tokens,
            'duration_tokens': duration_tokens
        }

    def extract_melody_tokens(self, melody_path):
        """
        Extract pitch and duration tokens from a MIDI file.
        """
        midi_data = pretty_midi.PrettyMIDI(melody_path)

        # Gather notes from all instruments (excluding drums)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch = note.pitch
                    # Duration quantization: you can discretize duration as needed.
                    # For simplicity, convert duration to an integer index in some range.
                    # Example: assume max duration ~6.3 sec mapped to 512 bins
                    # duration scaled to a bin index
                    duration = int((note.end - note.start) / 6.3 * 511)  # scale and clip if needed
                    duration = max(0, min(duration, 511))
                    notes.append((pitch, duration))

        # Sort notes by start time is recommended if you also consider note.start, but here we just trust the order given.
        # If needed, you can store start times and sort: notes.sort(key=lambda x: x_start)

        # Convert to tensors. If you have a maximum note length, trim or pad.
        # Let's assume a max length, e.g., N = 32 notes.
        N = 32
        if len(notes) > N:
            notes = notes[:N]

        # Pad if fewer than N notes
        while len(notes) < N:
            notes.append((0, 0))  # 0 indicates silence or padding

        pitches = [n[0] for n in notes]      # pitch range typically 0-127
        durations = [n[1] for n in notes]    # duration range scaled to 0-511

        pitch_tokens = torch.tensor(pitches, dtype=torch.long)
        duration_tokens = torch.tensor(durations, dtype=torch.long)
        return pitch_tokens, duration_tokens


###########################################
# Training Loop Example
###########################################


import os

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        audio_features = {k: v.to(device) for k, v in batch['audio_features'].items()}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pitch_tokens = batch['pitch_tokens']
        duration_tokens = batch['duration_tokens']

        optimizer.zero_grad()
        audio_emb, text_emb, melody_emb = model(
            audio_features,
            input_ids,
            attention_mask,
            pitch_tokens,
            duration_tokens
        )
        loss = criterion(audio_emb, text_emb, melody_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    tokenizer = processor.tokenizer

    melody_encoder = MelodyEncoder()
    clmp_model = CLMPModel(clap_model, melody_encoder, final_dim=512).to(device)

    dataset = AudioTextMelodyDataset(
        audio_dir="/root/m2music/data/audio",
        melody_dir="/root/m2music/data/melodies",
        captions_file="/root/m2music/data/captions.txt",
        processor=processor,
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=10)

    criterion = SymmetricTripleContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(clmp_model.parameters(), lr=3e-4)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(clmp_model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save checkpoint at the end of the epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"clmp_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': clmp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

    print("Training Completed")


if __name__ == "__main__":
    main()

import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi
import numpy as np
import argparse

# 1. Dataset Definition
class AudioTextMelodyDataset(Dataset):
    def __init__(self, audio_dir, melodies_dir, captions_file, processor, audio_transform=None, melody_transform=None):
        self.audio_dir = audio_dir
        self.melodies_dir = melodies_dir
        self.audio_transform = audio_transform
        self.melody_transform = melody_transform
        self.processor = processor

        # Load captions and filter out any empty lines
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]

        # Filter only .wav and .mid files and sort them
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.melody_files = sorted([f for f in os.listdir(melodies_dir) if f.endswith('.mid')])

        # Debugging prints
        print(f"[Dataset Init] Number of audio files: {len(self.audio_files)}")
        print(f"[Dataset Init] Number of melody files: {len(self.melody_files)}")
        print(f"[Dataset Init] Number of captions: {len(self.captions)}")

        # Verify counts match
        assert len(self.captions) == len(self.audio_files) == len(self.melody_files), (
            f"Data mismatch! "
            f"Number of audio files: {len(self.audio_files)}, "
            f"Number of melody files: {len(self.melody_files)}, "
            f"Number of captions: {len(self.captions)}"
        )

        # Optionally, print first few filenames and captions for verification
        print(f"First 5 audio files: {self.audio_files[:5]}")
        print(f"First 5 melody files: {self.melody_files[:5]}")
        print(f"First 5 captions: {self.captions[:5]}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            # Load and process audio
            audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
            waveform, sample_rate = torchaudio.load(audio_path)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)

            # Ensure the sample rate is as expected
            expected_sample_rate = self.processor.feature_extractor.sampling_rate
            if sample_rate != expected_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Process audio
            audio_features = self.processor.feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            )
            input_features = audio_features['input_features'].squeeze(0)  # Shape: (features, )

            # Load and process melody
            melody_path = os.path.join(self.melodies_dir, self.melody_files[idx])
            melody_embedding = self.extract_melody(melody_path)
            if self.melody_transform:
                melody_embedding = self.melody_transform(melody_embedding)

            # Process text
            caption = self.captions[idx]
            text_tokens = self.processor.tokenizer(
                caption,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            input_ids = text_tokens['input_ids'].squeeze(0)             # Shape: (max_length,)
            attention_mask = text_tokens['attention_mask'].squeeze(0)   # Shape: (max_length,)

            return {
                'input_features': input_features,
                'melody_embedding': melody_embedding,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise e

    def extract_melody(self, melody_path):
        """
        Extract a fixed-size melody representation from a MIDI file.
        This is a simplistic example. For better results, consider using a melody encoder (e.g., CNN, RNN).
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(melody_path)
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append((note.pitch, note.end - note.start))
            # Sort notes by start time
            notes = sorted(notes, key=lambda x: x[1])
            # Flatten and pad/truncate
            melody = []
            for pitch, duration in notes[:16]:  # Limit to first 16 notes
                melody.extend([pitch, duration])
            # Pad with zeros if less than 32 elements
            while len(melody) < 32:
                melody.append(0)
            # Truncate if more than 32 elements
            melody = melody[:32]
            # Convert to numpy array
            melody = np.array(melody, dtype=np.float32)
            return torch.tensor(melody, dtype=torch.float32)  # Shape: (32,)
        except Exception as e:
            print(f"Error processing melody file {melody_path}: {e}")
            raise e

# 2. CLaMP Model Definition
class CLaMP(nn.Module):
    def __init__(self, clap_model, embedding_dim=512):
        super(CLaMP, self).__init__()
        self.audio_model = clap_model.audio_model
        self.text_model = clap_model.text_model

        # Extract hidden sizes
        audio_hidden_size = self.audio_model.config.hidden_size
        text_hidden_size = self.text_model.config.hidden_size

        # Debugging statement
        print(f"Configuring projection layers with audio_hidden_size={audio_hidden_size} and text_hidden_size={text_hidden_size}")

        # Projection layers for audio and text
        self.audio_proj = nn.Linear(audio_hidden_size, embedding_dim)
        self.text_proj = nn.Linear(text_hidden_size, embedding_dim)

        # Projection layer for melody
        self.melody_proj = nn.Linear(32, embedding_dim)  # Assuming melody_embedding size is 32

    def forward(self, inputs):
        # Extract audio features
        audio_inputs = inputs['input_features']  # Shape: (batch_size, features)
        audio_outputs = self.audio_model(input_features=audio_inputs)
        if hasattr(audio_outputs, 'last_hidden_state') and audio_outputs.last_hidden_state is not None:
            print(f"audio_outputs.last_hidden_state shape: {audio_outputs.last_hidden_state.shape}")
            audio_features = audio_outputs.last_hidden_state  # Shape: (batch_size, hidden_size, ...)
            # Adjust pooling based on actual shape
            # Example pooling over last two dimensions (assuming shape [batch_size, hidden_size, 2, 32])
            audio_mean = audio_features.mean(dim=(-2, -1))  # Shape: (batch_size, hidden_size)
            print(f"audio_features.mean(dim=(-2, -1)) shape: {audio_mean.shape}")
            audio_embeddings = self.audio_proj(audio_mean)    # Shape: (batch_size, embedding_dim)
            print(f"audio_embeddings shape: {audio_embeddings.shape}")
        else:
            raise AttributeError("audio_model output does not have 'last_hidden_state' or it is None.")

        # Extract melody features
        melody_embedding = inputs['melody_embedding']  # Shape: (batch_size, 32)
        print(f"melody_embedding shape: {melody_embedding.shape}")
        melody_embeddings = self.melody_proj(melody_embedding)  # Shape: (batch_size, embedding_dim)
        print(f"melody_embeddings shape: {melody_embeddings.shape}")

        # Extract text features
        text_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        text_outputs = self.text_model(**text_inputs)
        if hasattr(text_outputs, 'last_hidden_state') and text_outputs.last_hidden_state is not None:
            print(f"text_outputs.last_hidden_state shape: {text_outputs.last_hidden_state.shape}")
            text_features = text_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
            text_mean = text_features.mean(dim=1)           # Shape: (batch_size, hidden_size)
            print(f"text_features.mean(dim=1) shape: {text_mean.shape}")
            text_embeddings = self.text_proj(text_mean)     # Shape: (batch_size, embedding_dim)
            print(f"text_embeddings shape: {text_embeddings.shape}")
        else:
            raise AttributeError("text_model output does not have 'last_hidden_state' or it is None.")

        # Combine embeddings (e.g., concatenate)
        combined_embeddings = torch.cat((audio_embeddings, melody_embeddings, text_embeddings), dim=1)  # Shape: (batch_size, embedding_dim * 3)
        print(f"combined_embeddings shape: {combined_embeddings.shape}")

        return combined_embeddings  # Modify as per your loss function

# 3. Triple Contrastive Loss Definition
class TripleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(TripleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, combined_embeddings):
        """
        Compute contrastive loss across three modalities: audio, melody, and text.

        Args:
            combined_embeddings (torch.Tensor): Shape (batch_size, embedding_dim * 3)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = combined_embeddings.size(0)
        embedding_dim = combined_embeddings.size(1) // 3

        # Split combined embeddings
        audio_emb = combined_embeddings[:, :embedding_dim]        # Shape: (batch_size, embedding_dim)
        melody_emb = combined_embeddings[:, embedding_dim:2*embedding_dim]  # Shape: (batch_size, embedding_dim)
        text_emb = combined_embeddings[:, 2*embedding_dim:]       # Shape: (batch_size, embedding_dim)

        # Normalize embeddings
        audio_norm = F.normalize(audio_emb, p=2, dim=1)
        melody_norm = F.normalize(melody_emb, p=2, dim=1)
        text_norm = F.normalize(text_emb, p=2, dim=1)

        # Compute similarity matrices
        sim_audio_melody = torch.matmul(audio_norm, melody_norm.T) / self.temperature  # Shape: (batch_size, batch_size)
        sim_audio_text = torch.matmul(audio_norm, text_norm.T) / self.temperature        # Shape: (batch_size, batch_size)
        sim_melody_text = torch.matmul(melody_norm, text_norm.T) / self.temperature      # Shape: (batch_size, batch_size)

        # Labels are diagonal since embeddings are paired
        labels = torch.arange(batch_size).to(combined_embeddings.device)

        # Compute cross-entropy loss for each pair
        loss_audio_melody = F.cross_entropy(sim_audio_melody, labels)
        loss_audio_text = F.cross_entropy(sim_audio_text, labels)
        loss_melody_text = F.cross_entropy(sim_melody_text, labels)

        # Average the losses
        loss = (loss_audio_melody + loss_audio_text + loss_melody_text) / 3
        return loss

# 4. Training Function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move all inputs to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        combined_embeddings = model(batch)  # Shape: (batch_size, embedding_dim * 3)

        # Compute loss
        loss = criterion(combined_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss

# 5. Main Training Loop
def main():
    parser = argparse.ArgumentParser(description="CLaMP Training with Audio, Text, and Melody")
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to the directory containing audio files')
    parser.add_argument('--melodies_dir', type=str, required=True, help='Path to the directory containing melody (MIDI) files')
    parser.add_argument('--captions_file', type=str, required=True, help='Path to the captions text file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the projection embeddings')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Verify that paths exist
    assert os.path.isdir(args.audio_dir), f"Audio directory does not exist: {args.audio_dir}"
    assert os.path.isdir(args.melodies_dir), f"Melodies directory does not exist: {args.melodies_dir}"
    assert os.path.isfile(args.captions_file), f"Captions file does not exist: {args.captions_file}"

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
    clap_model.to(device)

    # Inspect model attributes for debugging
    print("ClapModel attributes:", dir(clap_model))
    print("ClapModel:", clap_model)

    # Print hidden sizes for verification
    print(f"Audio model hidden size: {clap_model.audio_model.config.hidden_size}")
    print(f"Text model hidden size: {clap_model.text_model.config.hidden_size}")

    # Initialize Dataset and DataLoader
    print("Initializing dataset and dataloader...")
    dataset = AudioTextMelodyDataset(
        audio_dir=args.audio_dir,
        melodies_dir=args.melodies_dir,
        captions_file=args.captions_file,
        processor=processor,
        audio_transform=None,
        melody_transform=None  # Add any melody-specific transformations if needed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize CLaMP model
    print("Initializing CLaMP model...")
    model = CLaMP(clap_model, embedding_dim=args.embedding_dim).to(device)

    # Initialize loss function and optimizer
    print("Initializing loss function and optimizer...")
    criterion = TripleContrastiveLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(
        list(model.audio_proj.parameters()) +
        list(model.text_proj.parameters()) +
        list(model.melody_proj.parameters()),
        lr=args.learning_rate
    )

    # Training Loop
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"clamp_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint: {checkpoint_path}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()

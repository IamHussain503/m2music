import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import torch.nn as nn

# 1. Dataset Definition
class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, captions_file, processor, audio_transform=None):
        self.audio_dir = audio_dir
        self.audio_transform = audio_transform
        self.processor = processor

        # Load captions and filter out any empty lines
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]

        # Filter only .wav files and sort them
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

        # Debugging prints
        print(f"[Dataset Init] Number of audio files: {len(self.audio_files)}")
        print(f"[Dataset Init] Number of captions: {len(self.captions)}")

        # Verify counts match
        assert len(self.captions) == len(self.audio_files), (
            f"Audio and text mismatch! "
            f"Number of audio files: {len(self.audio_files)}, "
            f"Number of captions: {len(self.captions)}"
        )

        # Optionally, print first few filenames and captions for verification
        print(f"First 5 audio files: {self.audio_files[:5]}")
        print(f"First 5 captions: {self.captions[:5]}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            # Load audio
            audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
            waveform, sample_rate = torchaudio.load(audio_path)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)

            # Ensure the sample rate is as expected
            expected_sample_rate = self.processor.feature_extractor.sampling_rate
            if sample_rate != expected_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
                waveform = resampler(waveform)

            # Handle multi-channel audio by averaging to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Process audio using the feature extractor
            audio_features = self.processor.feature_extractor(
                waveform.squeeze(0).numpy(),
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            )
            input_features = audio_features['input_features'].squeeze(0)  # Shape: (features, )

            # Process text using the tokenizer
            caption = self.captions[idx]
            text_tokens = self.processor.tokenizer(
                caption,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            input_ids = text_tokens['input_ids'].squeeze(0)         # Shape: (max_length,)
            attention_mask = text_tokens['attention_mask'].squeeze(0)  # Shape: (max_length,)

            return {
                'input_features': input_features,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
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

        # Projection layers
        self.audio_proj = nn.Linear(audio_hidden_size, embedding_dim)
        self.text_proj = nn.Linear(text_hidden_size, embedding_dim)

    def forward(self, inputs):
        # Extract audio features
        audio_inputs = inputs['input_features']  # Shape: (batch_size, features)
        audio_outputs = self.audio_model(input_features=audio_inputs)
        if hasattr(audio_outputs, 'last_hidden_state') and audio_outputs.last_hidden_state is not None:
            print(f"audio_outputs.last_hidden_state shape: {audio_outputs.last_hidden_state.shape}")
            audio_features = audio_outputs.last_hidden_state  # Shape: (batch_size, hidden_size, 2,32)
            audio_mean = audio_features.mean(dim=(-2, -1))  # Shape: (batch_size, hidden_size)
            print(f"audio_features.mean(dim=(-2, -1)) shape: {audio_mean.shape}")
            audio_embeddings = self.audio_proj(audio_mean)  # Shape: (batch_size, embedding_dim)
            print(f"audio_embeddings shape: {audio_embeddings.shape}")
        else:
            raise AttributeError("audio_model output does not have 'last_hidden_state' or it is None.")

        # Extract text features
        text_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        text_outputs = self.text_model(**text_inputs)
        if hasattr(text_outputs, 'last_hidden_state') and text_outputs.last_hidden_state is not None:
            print(f"text_outputs.last_hidden_state shape: {text_outputs.last_hidden_state.shape}")
            text_features = text_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
            text_mean = text_features.mean(dim=1)  # Shape: (batch_size, hidden_size)
            print(f"text_features.mean(dim=1) shape: {text_mean.shape}")
            text_embeddings = self.text_proj(text_mean)    # Shape: (batch_size, embedding_dim)
            print(f"text_embeddings shape: {text_embeddings.shape}")
        else:
            raise AttributeError("text_model output does not have 'last_hidden_state' or it is None.")

        return audio_embeddings, text_embeddings

# 3. Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, text_embeddings):
        # Normalize embeddings
        audio_embeddings = nn.functional.normalize(audio_embeddings, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=1)

        # Compute similarities
        logits = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature  # Shape: (batch_size, batch_size)
        labels = torch.arange(len(logits)).to(logits.device)  # Shape: (batch_size,)

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

# 4. Training Function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Move all inputs to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        audio_embeddings, text_embeddings = model(batch)

        # Contrastive loss
        loss = criterion(audio_embeddings, text_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 5. Main Training Loop
def main():
    # Hyperparameters
    base_dir = "/root/m2music/clm_data"  # Ensure this is the correct absolute path
    audio_dir = os.path.join(base_dir, "audio")  # Assuming audio files are in 'audio/' subdirectory
    captions_file = os.path.join(base_dir, "captions.txt")
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Verify that paths exist
    assert os.path.isdir(audio_dir), f"Audio directory does not exist: {audio_dir}"
    assert os.path.isfile(captions_file), f"Captions file does not exist: {captions_file}"

    # Tokenizer, Processor, and Models
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

    # Dataset and Dataloader
    print("Initializing dataset and dataloader...")
    dataset = AudioTextDataset(audio_dir, captions_file, processor, audio_transform=None)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for debugging; increase after resolving issues
        pin_memory=True
    )

    # Model, Loss, Optimizer
    print("Initializing CLaMP model, loss function, and optimizer...")
    model = CLaMP(clap_model).to(device)
    criterion = ContrastiveLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.audio_proj.parameters()) + list(model.text_proj.parameters()),
        lr=learning_rate
    )

    # **Test with a Single Batch Before Full Training**
    print("Testing model with a single batch for debugging...")
    try:
        test_batch = next(iter(dataloader))
        test_batch = {k: v.to(device) for k, v in test_batch.items()}
        audio_emb, text_emb = model(test_batch)
        print(f"Test Audio Embeddings Shape: {audio_emb.shape}")
        print(f"Test Text Embeddings Shape: {text_emb.shape}")
    except Exception as e:
        print(f"Error during test batch processing: {e}")
        return

    # Training Loop
    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(base_dir, f"clamp_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()

import os
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# 1. CLaMP Model Definition
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
            audio_features = audio_outputs.last_hidden_state  # Shape: (batch_size, hidden_size, ...)
            # Adjust pooling based on the actual shape
            # Example pooling over last two dimensions
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

# 2. Contrastive Loss Definition (if needed for further evaluation)
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, text_embeddings):
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature  # Shape: (batch_size, batch_size)
        labels = torch.arange(len(logits)).to(logits.device)  # Shape: (batch_size,)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

# 3. Prediction Function
def predict(model, processor, audio_path, text):
    """
    Generate embeddings for a given audio file and text input.

    Args:
        model (nn.Module): The trained CLaMP model.
        processor (AutoProcessor): The processor for handling inputs.
        audio_path (str): Path to the audio file (.wav format).
        text (str): Text input.

    Returns:
        tuple: (audio_embedding, text_embedding)
    """
    model.eval()

    # Load and process audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if processor.feature_extractor.sampling_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=processor.feature_extractor.sampling_rate)
        waveform = resampler(waveform)

    # Handle multi-channel audio by averaging to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Process audio
    with torch.no_grad():
        audio_features = processor.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt"
        )
    input_features = audio_features['input_features'].squeeze(0)  # Shape: (features, )

    # Process text
    with torch.no_grad():
        text_tokens = processor.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    input_ids = text_tokens['input_ids'].squeeze(0)         # Shape: (max_length,)
    attention_mask = text_tokens['attention_mask'].squeeze(0)  # Shape: (max_length,)

    # Create input dictionary
    inputs = {
        'input_features': input_features.unsqueeze(0).to(model.audio_model.device),  # Shape: (1, features)
        'input_ids': input_ids.unsqueeze(0).to(model.text_model.device),             # Shape: (1, max_length)
        'attention_mask': attention_mask.unsqueeze(0).to(model.text_model.device)    # Shape: (1, max_length)
    }

    # Generate embeddings
    with torch.no_grad():
        audio_emb, text_emb = model(inputs)

    return audio_emb.squeeze(0), text_emb.squeeze(0)

# 4. Similarity Computation Function
def compute_cosine_similarity(audio_emb, text_emb):
    """
    Compute cosine similarity between audio and text embeddings.

    Args:
        audio_emb (torch.Tensor): Audio embedding vector.
        text_emb (torch.Tensor): Text embedding vector.

    Returns:
        float: Cosine similarity score.
    """
    cosine_sim = F.cosine_similarity(audio_emb, text_emb, dim=0).item()
    return cosine_sim

# 5. Main Function with Argument Parsing
def main():
    parser = argparse.ArgumentParser(description="CLaMP Model Testing and Prediction")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (e.g., clamp_epoch_10.pth)')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file for prediction (e.g., new_audio.wav)')
    parser.add_argument('--text', type=str, required=True, help='Text input for prediction')
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Load model
    print("Loading model...")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
    model = CLaMP(clap_model).to(device)

    # Load checkpoint
    print(f"Loading model checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Perform prediction
    print("Performing prediction...")
    audio_emb, text_emb = predict(model, processor, args.audio, args.text)
    print("Prediction completed.")

    # Compute cosine similarity
    similarity = compute_cosine_similarity(audio_emb, text_emb)
    print(f"Cosine Similarity between Audio and Text: {similarity:.4f}")

    # (Optional) Save Embeddings
    # torch.save({'audio_embedding': audio_emb, 'text_embedding': text_emb}, 'embeddings.pth')
    # print("Embeddings saved to embeddings.pth")

if __name__ == "__main__":
    main()

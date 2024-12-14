import os
import torch
import torchaudio
import pretty_midi
import numpy as np
from transformers import AutoModel, AutoProcessor
from main_clm import CLaMP  # Assuming the class definitions are in melody.py or adjust accordingly
import argparse

def extract_melody(melody_path):
    """
    Same melody extraction function used during training.
    """
    midi_data = pretty_midi.PrettyMIDI(melody_path)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.pitch, note.end - note.start))
    # Sort by start time
    notes = sorted(notes, key=lambda x: x[1])

    melody = []
    for pitch, duration in notes[:16]:
        melody.extend([pitch, duration])

    while len(melody) < 32:
        melody.append(0)
    melody = melody[:32]
    melody = np.array(melody, dtype=np.float32)
    return torch.tensor(melody, dtype=torch.float32)

def load_and_process_inputs(processor, audio_path, melody_path, caption):
    """
    Prepare the audio, melody, and text inputs for the model.
    """
    # Audio
    waveform, sample_rate = torchaudio.load(audio_path)
    expected_sample_rate = processor.feature_extractor.sampling_rate
    if sample_rate != expected_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_features = processor.feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=expected_sample_rate,
        return_tensors="pt"
    )
    input_features = audio_features['input_features'].squeeze(0)

    # Melody
    melody_embedding = extract_melody(melody_path)

    # Text
    text_tokens = processor.tokenizer(
        caption,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    input_ids = text_tokens['input_ids'].squeeze(0)
    attention_mask = text_tokens['attention_mask'].squeeze(0)

    batch = {
        'input_features': input_features.unsqueeze(0),  # Add batch dimension
        'melody_embedding': melody_embedding.unsqueeze(0),  # Add batch dimension
        'input_ids': input_ids.unsqueeze(0),
        'attention_mask': attention_mask.unsqueeze(0)
    }

    return batch

def main():
    parser = argparse.ArgumentParser(description="Inference with CLaMP model")
    parser.add_argument('--audio_path', type=str, required=True, help='Path to a single audio (.wav) file')
    parser.add_argument('--melody_path', type=str, required=True, help='Path to a single melody (.mid) file')
    parser.add_argument('--caption', type=str, required=True, help='Text caption describing the audio')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the saved model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    # Initialize the model
    model = CLaMP(clap_model, embedding_dim=512).to(device)

    # Load checkpoint
    print(f"Loading model checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device,weights_only=True)
    model.load_state_dict(state_dict,strict=False)
    model.eval()

    # Prepare inputs
    batch = load_and_process_inputs(processor, args.audio_path, args.melody_path, args.caption)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Run inference
    with torch.no_grad():
        embeddings = model(batch)  # Shape: (1, embedding_dim * 3)

    print("Embeddings shape:", embeddings.shape)
    print("Embeddings:", embeddings)

    # You can now use the embeddings for downstream tasks like similarity, retrieval, etc.

if __name__ == '__main__':
    main()


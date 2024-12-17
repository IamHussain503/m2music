import os
import torch
import torchaudio
import pretty_midi
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import argparse


class AudioTextMelodyDataset(Dataset):
    def __init__(self, audio_dir, melodies_dir, captions_file, processor):
        self.audio_dir = audio_dir
        self.melodies_dir = melodies_dir
        self.processor = processor

        # Load captions
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]

        # Get sorted list of audio and melody files
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.melody_files = sorted([f for f in os.listdir(melodies_dir) if f.endswith('.mid')])

        # Check consistency
        assert len(self.captions) == len(self.audio_files) == len(self.melody_files), "Mismatch in counts!"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        melody_path = os.path.join(self.melodies_dir, self.melody_files[idx])
        caption = self.captions[idx]

        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_path)
        expected_sample_rate = self.processor.feature_extractor.sampling_rate
        if sample_rate != expected_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_features = self.processor.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=expected_sample_rate,
            return_tensors="pt"
        )
        input_features = audio_features['input_features'].squeeze(0)

        # Melody
        melody_embedding = self.extract_melody(melody_path)

        # Text
        text_tokens = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        input_ids = text_tokens['input_ids'].squeeze(0)
        attention_mask = text_tokens['attention_mask'].squeeze(0)

        return {
            'input_features': input_features,
            'melody_embedding': melody_embedding,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def extract_melody(self, melody_path):
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


class CLaMP(nn.Module):
    def __init__(self, clap_model, embedding_dim=512):
        super(CLaMP, self).__init__()
        self.audio_model = clap_model.audio_model
        self.text_model = clap_model.text_model

        # Extract hidden sizes
        audio_hidden_size = self.audio_model.config.hidden_size
        text_hidden_size = self.text_model.config.hidden_size

        # Projection layers for audio and text
        self.audio_proj = nn.Linear(audio_hidden_size, embedding_dim)
        self.text_proj = nn.Linear(text_hidden_size, embedding_dim)

        # Melody projection layer (matches checkpoint structure)
        self.melody_proj = nn.Linear(32, embedding_dim)

    def forward(self, inputs):
        # Audio features
        audio_inputs = inputs['input_features']
        audio_outputs = self.audio_model(input_features=audio_inputs)
        audio_features = audio_outputs.last_hidden_state.mean(dim=(-2, -1))
        audio_embeddings = self.audio_proj(audio_features)

        # Melody features
        melody_input = inputs['melody_embedding']  # Shape: (batch_size, 32)
        melody_embeddings = self.melody_proj(melody_input)

        # Text features
        text_inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
        text_outputs = self.text_model(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_embeddings = self.text_proj(text_features)

        # Combine embeddings
        combined_embeddings = torch.cat((audio_embeddings, melody_embeddings, text_embeddings), dim=1)
        return combined_embeddings


def main():
    parser = argparse.ArgumentParser(description="Batch Embeddings Extraction")
    parser.add_argument('--audio_path', type=str, required=True, help='Path to directory of .wav files')
    parser.add_argument('--melody_path', type=str, required=True, help='Path to directory of .mid files')
    parser.add_argument('--caption', type=str, required=True, help='Path to captions.txt')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    model = CLaMP(clap_model, embedding_dim=512).to(device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    dataset = AudioTextMelodyDataset(args.audio_path, args.melody_path, args.caption, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    audio_embeddings_list = []
    melody_embeddings_list = []
    text_embeddings_list = []

    embedding_dim = 512

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings = model(batch)  # Shape: (B, 1536)

            # Split embeddings
            audio_emb = embeddings[:, :embedding_dim].cpu().numpy()
            melody_emb = embeddings[:, embedding_dim:2*embedding_dim].cpu().numpy()
            text_emb = embeddings[:, 2*embedding_dim:].cpu().numpy()

            audio_embeddings_list.append(audio_emb)
            melody_embeddings_list.append(melody_emb)
            text_embeddings_list.append(text_emb)

    # Save embeddings
    audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
    melody_embeddings = np.concatenate(melody_embeddings_list, axis=0)
    text_embeddings = np.concatenate(text_embeddings_list, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "audio_embeddings.npy"), audio_embeddings)
    np.save(os.path.join(args.output_dir, "melody_embeddings.npy"), melody_embeddings)
    np.save(os.path.join(args.output_dir, "text_embeddings.npy"), text_embeddings)

    print("Embeddings saved successfully!")


if __name__ == '__main__':
    main()



# import os
# import torch
# import torchaudio
# import pretty_midi
# import numpy as np
# import torch.nn as nn
# from transformers import AutoModel, AutoProcessor
# from torch.utils.data import Dataset, DataLoader
# import argparse

# class AudioTextMelodyDataset(Dataset):
#     def __init__(self, audio_dir, melodies_dir, captions_file, processor):
#         self.audio_dir = audio_dir
#         self.melodies_dir = melodies_dir
#         self.processor = processor

#         # Load captions
#         with open(captions_file, 'r', encoding='utf-8') as f:
#             self.captions = [line.strip() for line in f if line.strip()]

#         # Get sorted list of audio and melody files
#         self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
#         self.melody_files = sorted([f for f in os.listdir(melodies_dir) if f.endswith('.mid')])

#         # Check consistency
#         assert len(self.captions) == len(self.audio_files) == len(self.melody_files), "Mismatch in counts!"

#     def __len__(self):
#         return len(self.audio_files)

#     def __getitem__(self, idx):
#         audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
#         melody_path = os.path.join(self.melodies_dir, self.melody_files[idx])
#         caption = self.captions[idx]

#         # Load and process audio
#         waveform, sample_rate = torchaudio.load(audio_path)
#         expected_sample_rate = self.processor.feature_extractor.sampling_rate
#         if sample_rate != expected_sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
#             waveform = resampler(waveform)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)
#         audio_features = self.processor.feature_extractor(
#             waveform.squeeze(0).numpy(),
#             sampling_rate=expected_sample_rate,
#             return_tensors="pt"
#         )
#         input_features = audio_features['input_features'].squeeze(0)

#         # Melody
#         melody_embedding = self.extract_melody(melody_path)

#         # Text
#         text_tokens = self.processor.tokenizer(
#             caption,
#             return_tensors="pt",
#             truncation=True,
#             padding="max_length",
#             max_length=128,
#         )
#         input_ids = text_tokens['input_ids'].squeeze(0)
#         attention_mask = text_tokens['attention_mask'].squeeze(0)

#         return {
#             'input_features': input_features,
#             'melody_embedding': melody_embedding,
#             'input_ids': input_ids,
#             'attention_mask': attention_mask
#         }

#     def extract_melody(self, melody_path):
#         midi_data = pretty_midi.PrettyMIDI(melody_path)
#         notes = []
#         for instrument in midi_data.instruments:
#             if not instrument.is_drum:
#                 for note in instrument.notes:
#                     notes.append((note.pitch, note.end - note.start))
#         # Sort by start time
#         notes = sorted(notes, key=lambda x: x[1])

#         melody = []
#         for pitch, duration in notes[:16]:
#             melody.extend([pitch, duration])

#         while len(melody) < 32:
#             melody.append(0)
#         melody = melody[:32]
#         melody = np.array(melody, dtype=np.float32)
#         return torch.tensor(melody, dtype=torch.float32)

# class CLaMP(nn.Module):
#     def __init__(self, clap_model, embedding_dim=512):
#         super(CLaMP, self).__init__()
#         self.audio_model = clap_model.audio_model
#         self.text_model = clap_model.text_model

#         # Extract hidden sizes
#         audio_hidden_size = self.audio_model.config.hidden_size
#         text_hidden_size = self.text_model.config.hidden_size

#         # Debugging statement
#         print(f"Configuring projection layers with audio_hidden_size={audio_hidden_size} and text_hidden_size={text_hidden_size}")

#         # Projection layers for audio and text
#         self.audio_proj = nn.Linear(audio_hidden_size, embedding_dim)
#         self.text_proj = nn.Linear(text_hidden_size, embedding_dim)

#         # Projection layer for melody
#         self.melody_proj = nn.Linear(32, embedding_dim)  # Assuming melody_embedding size is 32

#     def forward(self, inputs):
#         # Extract audio features
#         audio_inputs = inputs['input_features']  # Shape: (batch_size, features)
#         audio_outputs = self.audio_model(input_features=audio_inputs)
#         if hasattr(audio_outputs, 'last_hidden_state') and audio_outputs.last_hidden_state is not None:
#             print(f"audio_outputs.last_hidden_state shape: {audio_outputs.last_hidden_state.shape}")
#             audio_features = audio_outputs.last_hidden_state  # Shape: (batch_size, hidden_size, ...)
#             # Adjust pooling based on actual shape
#             # Example pooling over last two dimensions (assuming shape [batch_size, hidden_size, 2, 32])
#             audio_mean = audio_features.mean(dim=(-2, -1))  # Shape: (batch_size, hidden_size)
#             print(f"audio_features.mean(dim=(-2, -1)) shape: {audio_mean.shape}")
#             audio_embeddings = self.audio_proj(audio_mean)    # Shape: (batch_size, embedding_dim)
#             print(f"audio_embeddings shape: {audio_embeddings.shape}")
#         else:
#             raise AttributeError("audio_model output does not have 'last_hidden_state' or it is None.")

#         # Extract melody features
#         melody_embedding = inputs['melody_embedding']  # Shape: (batch_size, 32)
#         print(f"melody_embedding shape: {melody_embedding.shape}")
#         melody_embeddings = self.melody_proj(melody_embedding)  # Shape: (batch_size, embedding_dim)
#         print(f"melody_embeddings shape: {melody_embeddings.shape}")

#         # Extract text features
#         text_inputs = {
#             'input_ids': inputs['input_ids'],
#             'attention_mask': inputs['attention_mask']
#         }
#         text_outputs = self.text_model(**text_inputs)
#         if hasattr(text_outputs, 'last_hidden_state') and text_outputs.last_hidden_state is not None:
#             print(f"text_outputs.last_hidden_state shape: {text_outputs.last_hidden_state.shape}")
#             text_features = text_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
#             text_mean = text_features.mean(dim=1)           # Shape: (batch_size, hidden_size)
#             print(f"text_features.mean(dim=1) shape: {text_mean.shape}")
#             text_embeddings = self.text_proj(text_mean)     # Shape: (batch_size, embedding_dim)
#             print(f"text_embeddings shape: {text_embeddings.shape}")
#         else:
#             raise AttributeError("text_model output does not have 'last_hidden_state' or it is None.")

#         # Combine embeddings (e.g., concatenate)
#         combined_embeddings = torch.cat((audio_embeddings, melody_embeddings, text_embeddings), dim=1)  # Shape: (batch_size, embedding_dim * 3)
#         print(f"combined_embeddings shape: {combined_embeddings.shape}")

#         return combined_embeddings  # Modify as per your loss function

# def main():
#     parser = argparse.ArgumentParser(description="Batch Embeddings Extraction")
#     parser.add_argument('--audio_path', type=str, required=True, help='Path to directory of .wav files')
#     parser.add_argument('--melody_path', type=str, required=True, help='Path to directory of .mid files')
#     parser.add_argument('--caption', type=str, required=True, help='Path to captions.txt')
#     parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
#     parser.add_argument('--output_dir', type=str, default='.', help='Output directory for embeddings')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
#     clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)

#     model = CLaMP(clap_model, embedding_dim=512).to(device)
#     state_dict = torch.load(args.checkpoint, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()

#     dataset = AudioTextMelodyDataset(args.audio_path, args.melody_path, args.caption, processor)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

#     audio_embeddings_list = []
#     melody_embeddings_list = []
#     text_embeddings_list = []

#     embedding_dim = 512

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             embeddings = model(batch)  # (B, 1536)

#             # Split embeddings
#             audio_emb = embeddings[:, :embedding_dim].cpu().numpy()
#             melody_emb = embeddings[:, embedding_dim:2*embedding_dim].cpu().numpy()
#             text_emb = embeddings[:, 2*embedding_dim:].cpu().numpy()

#             audio_embeddings_list.append(audio_emb)
#             melody_embeddings_list.append(melody_emb)
#             text_embeddings_list.append(text_emb)

#     audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
#     melody_embeddings = np.concatenate(melody_embeddings_list, axis=0)
#     text_embeddings = np.concatenate(text_embeddings_list, axis=0)

#     os.makedirs(args.output_dir, exist_ok=True)
#     np.save(os.path.join(args.output_dir, "audio_embeddings.npy"), audio_embeddings)
#     np.save(os.path.join(args.output_dir, "melody_embeddings.npy"), melody_embeddings)
#     np.save(os.path.join(args.output_dir, "text_embeddings.npy"), text_embeddings)

#     print("Embeddings saved successfully!")

# if __name__ == '__main__':
#     main()

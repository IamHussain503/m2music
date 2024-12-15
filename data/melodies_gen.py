import os
import subprocess

# Set your input and output directories
input_dir = "/root/m2music/data/audio"
output_dir = "/root/m2music/data/melodies"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all WAV files in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".wav"):
        input_path = os.path.join(input_dir, filename)
        
        # Run the basic-pitch command
        # Adjust the order of arguments if required by your environment
        # The user requested format: basic-pitch /output/directory/path /input/audio/path
        subprocess.run(["basic-pitch", output_dir, input_path], check=True)
        
        print(f"Processed: {input_path} -> {output_dir}")
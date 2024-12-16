import os

directory = "/root/m2music/data/melodies"
for filename in os.listdir(directory):
    if filename.endswith("_basic_pitch.mid"):
        new_name = filename.replace("_basic_pitch", "")
        os.rename(os.path.join(directory, filename),
                  os.path.join(directory, new_name))
        print(f"Renamed {filename} to {new_name}")
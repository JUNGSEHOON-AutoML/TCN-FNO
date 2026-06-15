import os
print("Searching entire userhome4 for wav files...")
exclude = {".cache", ".local", "miniconda3", "transformers", ".npm", ".nvm"}
for root, dirs, files in os.walk("/userHome/userhome4"):
    dirs[:] = [d for d in dirs if d not in exclude]
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            print(path, f"{os.path.getsize(path)/(1024*1024):.2f} MB")
            break
print("Search finished.")

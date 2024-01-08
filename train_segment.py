import subprocess

# Replace the paths below with the actual paths to your scripts
train_script_path = "train.py"
segment_script_path = "segment_stardist.py"

# Run train.py
print("Running train.py...")
train_process = subprocess.run(["python", train_script_path])

# Check the exit code of train.py
if train_process.returncode != 0:
    print("train.py failed. Exiting.")
    exit(train_process.returncode)

# Run segment_stardist.py
print("Running segment_stardist.py...")
segment_process = subprocess.run(["python", segment_script_path])

# Check the exit code of segment_stardist.py
if segment_process.returncode != 0:
    print("segment_stardist.py failed. Exiting.")
    exit(segment_process.returncode)

print("Python script completed successfully.")
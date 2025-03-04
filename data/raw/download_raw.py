### Download original datasets ###

import gdown 
import zipfile
import os


# Download file

print("Downloading raw.zip")
output = "raw.zip"
gdown.download(f"https://drive.google.com/file/d/1Fw0FgMi7D35YdWLazLv8I25j4oYxODNR/view?usp=sharing", output, fuzzy = True, quiet = False)


# Extract zip file

print("Unzipping file...")

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall(".")

#Delete the zipfile after extracting itconda 

os.remove(output)


print("Download and extraction completed")
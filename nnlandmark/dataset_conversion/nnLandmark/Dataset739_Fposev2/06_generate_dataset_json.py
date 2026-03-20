import json, os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnlandmark.dataset_conversion.generate_dataset_json import generate_dataset_json

# ------------------------------------------------------------------ paths
root = Path("/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset739_Fposev2")
imagesTr = root / "imagesTr"
imagesTs = root / "imagesTs"

# ------------------------------------------------------------------ derive name→label from landmarks
all_landmarks = json.loads((root / "all_landmarks_voxel.json").read_text())
name2label = {}
for case_lms in all_landmarks.values():
    for name in case_lms:
        if name not in name2label:
            name2label[name] = len(name2label) + 1
labels = {"background": 0, **name2label}

print(f"{len(labels)-1} foreground labels loaded")
# e.g. {0:'background', 1:'landmark_1', 2:'landmark_2', …}

# ------------------------------------------------------------------ write dataset.json
generate_dataset_json(
    output_folder=root,
    channel_names=({0: 'US/MR'}),           
    labels=labels,  
    num_training_cases=len(os.listdir(imagesTr)),
    file_ending=".nii.gz", 
    dataset_name="Dataset739_Fposev2",                    # human-readable or task ID
    license="hands off!"
)

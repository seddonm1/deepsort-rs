import numpy as np
import os

# script to convert the deep_sort pregenerated detections from https://drive.google.com/open?id=1VVqtL0klSUvLnmBKS89il1EKC3IxUBVK
# into just the raw det.txt format (strip the feature vector)
for filename in os.listdir("./resources/detections/MOT16-train/"):
    print(filename)
    data = np.load(f"./resources/detections/MOT16-train/{filename}")
    sequence = filename.replace(".npy", "")
    with open(f"../TrackEval/data/gt/mot_challenge/MOT16-train/{sequence}/det/deepsort_det.txt", 'w') as f:
        for row in data:
            f.write(f"{int(row[0])},{int(row[1])},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{int(row[7])},{int(row[8])},{int(row[9])}\n")
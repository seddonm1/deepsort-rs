find .. -name '.DS_Store' -type f -delete &&\
time python3 evaluate_motchallenge.py \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT16-train \
--detection_dir=./resources/detections/MOT16-train \
--output=../TrackEval/data/trackers/mot_challenge/MOT16-train/deep_sort/data \
--min_confidence=-100.0 \
--min_detection_height=0 \
--nms_max_overlap=0.5 \
--max_cosine_distance=0.2 \
--nn_budget=100
find .. -name '.DS_Store' -type f -delete &&\
time python3 evaluate_motchallenge.py \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT16-train \
--detection_dir=./resources/detections/MOT16-train \
--output=../TrackEval/data/trackers/mot_challenge/MOT16-train/deep_sort/data \
--min_confidence=0.3 \
--min_detection_height=0 \
--nms_max_overlap=1.0 \
--max_cosine_distance=0.2 \
--nn_budget=100
time python3 evaluate_motchallenge.py \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT16-test \
--detection_dir=./resources/detections/MOT16-test \
--output=../TrackEval/data/trackers/mot_challenge/MOT16-test/deep_sort/data \
--min_confidence=0.3 \
--min_detection_height=0 \
--nms_max_overlap=1.0 \
--max_cosine_distance=0.2 \
--nn_budget=100
time python3 evaluate_motchallenge.py \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT17-train \
--detection_dir=./resources/detections/MOT17-train \
--output=../TrackEval/data/trackers/mot_challenge/MOT17-train/deep_sort/data \
--min_confidence=0.3 \
--min_detection_height=0 \
--nms_max_overlap=1.0 \
--max_cosine_distance=0.2 \
--nn_budget=100
time python3 evaluate_motchallenge.py \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT17-test \
--detection_dir=./resources/detections/MOT17-test \
--output=../TrackEval/data/trackers/mot_challenge/MOT17-test/deep_sort/data \
--min_confidence=0.3 \
--min_detection_height=0 \
--nms_max_overlap=1.0 \
--max_cosine_distance=0.2 \
--nn_budget=100
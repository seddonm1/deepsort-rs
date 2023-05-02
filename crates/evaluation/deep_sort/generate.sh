find .. -name '.DS_Store' -type f -delete &&\
python3 tools/generate_detections.py \
--model=./mars-small128.pb \
--mot_dir=../TrackEval/data/gt/mot_challenge/MOT16-train \
--output_dir=./resources/detections/MOT16-train
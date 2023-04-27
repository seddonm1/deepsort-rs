ffmpeg \
-framerate 30 \
-i out/%6d.jpg \
-c:v libx264 \
-pix_fmt yuv420p \
-r 30 \
video.mp4
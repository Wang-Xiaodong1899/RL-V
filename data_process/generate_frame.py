import os
import subprocess
from tqdm import tqdm

video_folder = "/mnt/storage/user/wangxiaodong/RLAIF-V/data_process/dataset/train"
output_folder = "/mnt/storage/user/wangxiaodong/RLAIF-V/data_process/dataset/frames"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for video_file in tqdm(os.listdir(video_folder)):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_output_folder = os.path.join(output_folder, video_name)

        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)
        else:
            print(f'{video_name} already has frames')
            continue

        ffmpeg_command = [
            "ffmpeg",
            "-loglevel", "quiet",
            "-i", video_path,
            "-vf", "scale=336:-1,fps=2",
            os.path.join(frame_output_folder, "frame_%04d.png")
        ]

        subprocess.run(ffmpeg_command)

print("Done!")

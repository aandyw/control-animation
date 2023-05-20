import os


# App Pose utils
def motion_to_video_path(motion):
    videos = [
        "__assets__/walk_01.mp4",
        "__assets__/walk_02.mp4",
        "__assets__/walk_03.mp4",
        "__assets__/run.mp4",
        "__assets__/dance1_corr.mp4",
        "__assets__/dance2_corr.mp4",
        "__assets__/dance3_corr.mp4",
        "__assets__/dance4_corr.mp4",
        "__assets__/dance5_corr.mp4",
    ]
    if len(motion.split(" ")) > 1 and motion.split(" ")[1].isnumeric():
        id = int(motion.split(" ")[1]) - 1
        return videos[id]
    else:
        return motion

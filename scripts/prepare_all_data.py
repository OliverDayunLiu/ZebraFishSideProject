from subprocess import check_output
import os


data_dir = '../data/raw_video'
for filename in os.listdir(data_dir): #50mM dimethylone #1.avi
    if '.avi' not in filename:
        continue
    video = os.path.join(data_dir, filename) # ../data/raw_video/50mM dimethylone #1.avi
    img_dir = os.path.join(data_dir, filename.split('.')[0]) #../data/raw_video/50mM dimethylone #1
    if os.path.exists(img_dir):
        continue
    else:
        os.mkdir(img_dir)
        video = "\"" + video + "\""
        output_path = img_dir + '/%05d.png'
        output_path = "\"" + output_path + "\""
        cmd = 'ffmpeg -i ' + video + ' -r 25 ' + output_path
    check_output(cmd, shell=True).decode()
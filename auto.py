import subprocess
import argparse
import glob
import utils
import json
import hashlib

openpose_bin = 'asd'
feature_extractor = 'ss'
track_process = 'asdf'

parser = argparse.ArgumentParser()
parser.add_argument("video", help="path to video folder")
parser.add_argument("output", help="path to save output")
args = parser.parse_args()

video_path = args.video
output_path = args.output

video_list = glob.glob(video_path)

result = {'hash': '', 'status': []}


def sha256(text):
    m = hashlib.sha256()
    m.update(text)
    return m.hexdigest()


if len(video_list):
    utils.make_directory("%s/" % output_path)

    hashed = sha256(''.join(video_list))

    print hashed





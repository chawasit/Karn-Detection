import argparse
import glob
import hashlib
import json
import os
import subprocess
from os.path import exists, basename, join

import utils

openpose_path = join(os.getcwd(), '..', 'openpose')
feature_extractor = 'task2.py'
track_process = 'asdf'

parser = argparse.ArgumentParser()
parser.add_argument("video", help="path to video folder")
parser.add_argument("output", help="path to save output")
args = parser.parse_args()

video_path = args.video
output_path = args.output
result_json = "%s/result.json" % output_path
video_list = glob.glob("%s/*" % video_path)

result = {'hash': '', 'works': []}


def sha256(text):
    m = hashlib.sha256()
    m.update(text)
    return m.hexdigest()


if len(video_list):
    utils.make_directory("%s/" % output_path)
    result['hash'] = sha256(''.join(video_list))

    for video in video_list:
        result['works'].append(
            {'video': video, '1_openpose': False, '2_feature_extractor': False, '3_tracking': False})

    if exists(result_json):
        with open(result_json, 'r') as f:
            saved_result = json.loads(f.read())
            if saved_result['hash'] == result['hash']:
                result = saved_result

    try:
        for work in result['works']:
            video_path = work['video']
            video_name = basename(video_path)

            print "=== " + video_name + "==="

            base_output_path = "%s/%s" % (output_path, video_name)
            utils.make_directory(base_output_path)

            openpose_output_path = "%s/openpose" % base_output_path
            extractor_output_path = "%s/extractor" % base_output_path
            tracking_output_path = "%s/tracking" % base_output_path

            try:
                if not work['1_openpose']:
                    print "- [Estimating pose]"
                    print subprocess.call(
                        ['cd', openpose_path, '&&', '.\\bin\\OpenPoseDemo.exe', '-video', video_path, '-write_json', openpose_output_path,
                         '-render_pose', '0', '--no_display'], shell=True)

                    work['1_openpose'] = True

                if not work['2_feature_extractor']:
                    print "- [Extracting]"
                    subprocess.call(['python', feature_extractor, video_path, openpose_output_path, extractor_output_path], shell=True)

                    work['2_feature_extractor'] = True

                if not work['3_tracking']:
                    print "- [Tracking]"
                    subprocess.call(['exit', '0'])

                    work['3_tracking'] = True
            except Exception as e:
                print e
                continue

    finally:
        with open(result_json, 'w') as f:
            f.write(json.dumps(result, indent=2, sort_keys=True))

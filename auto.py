import argparse
import glob
import hashlib
import json
import os
import subprocess
from os.path import exists, basename, join
import shutil
import utils

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

openpose_path = join(os.getcwd(), '..', 'openpose')
feature_extractor = 'task2.py'
track_process = 'task3.py'

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

    final_result_output_path = "%s/final" % output_path
    utils.make_directory(final_result_output_path)
    final_result = open(final_result_output_path + "/20p32n0075.txt", 'w')

    try:
        for work in result['works']:
            video_path = work['video']
            video_name = basename(video_path).split('.')[0]

            print "=== " + video_name + "==="

            base_output_path = "%s/%s" % (output_path, video_name)
            utils.make_directory(base_output_path)

            openpose_output_path = "%s/openpose" % base_output_path
            extractor_output_path = "%s/extractor" % base_output_path
            tracking_output_path = "%s/tracking" % base_output_path
            final_output_path = "%s/final/%s" % (output_path, video_name)

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
                    subprocess.call(['python', track_process, extractor_output_path, tracking_output_path], shell=True)
                    utils.make_directory(final_output_path)
                    copytree(tracking_output_path, final_output_path)
                    work['3_tracking'] = True
                
                if work['1_openpose'] and work['2_feature_extractor'] and work['3_tracking']:
                    final_result.write(video_name+":"+str(len(glob.glob("%s/*" % final_output_path)))+"\n")

            except Exception as e:
                print e
                continue

    finally:
        final_result.close()

        with open(result_json, 'w') as f:
            f.write(json.dumps(result, indent=2, sort_keys=True))

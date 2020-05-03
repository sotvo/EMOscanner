from detector import *

detection_model_path = 'Emotion-recognition/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'Emotion-recognition/models/_mini_XCEPTION.102-0.66.hdf5'

'''
img_dir путь к папке, куда сохранять картинки из видео. если None, будет по вебке
video_path путь к видео, если анализируем видео. если None, будет по вебке
'''

img_dir, video_path = None, None # 'ims', 'vid2.mp4'

vd = VideoDetector(detection_model_path, emotion_model_path, img_dir)

'''
запустить, если есть видео
'''
#vd.extract_images(video_path) 

vd.annotate_imgs()
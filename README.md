# EMOscanner
Здесь пока ссылки выкладываем в основной ветке.

### quaser

https://github.com/antoinelame/GazeTracking  -непроверено

https://github.com/gkaguirrelab/transparentTrack -непроверенно

## Определение эмоций и движений глаз
### Инструкция по установке через Anaconda

1. Создать виртуальную среду с python=3.6  
conda create -c anaconda -n /env-name/ python=3.6 

2. Активировать среду  
conda activate /env-name/

3. Установить основные пакеты  
numpy
opencv-python
tensorflow  
keras  
ktrain

4. Установить библиотеку dlib == 19.16.0 следующим образом:  
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f

5. Запуск:  
python test_detect.py ИЛИ запустить test_detect.ipynb

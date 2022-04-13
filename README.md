# Diplom

# Edge Processing - Aircraft Detection

Aircraft Detection with local storage (sqlite3)


1) Install all necessary dependencies:  
    ```
    sudo apt install libopencv-dev python3-opencv
    pip3 install -r requirements.txt 
    pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
    ```
2) To simulate the RTMP stream from a video file, it is also necessary to set the nginx-rtmp-module. Link to the installation guide: https://github.com/arut/nginx-rtmp-module.
```
gst-launch-1.0 -v filesrc location=ten_minutes.mp4 ! decodebin ! queue ! videoconvert !  omxh264enc ! flvmux ! rtmpsink location=rtmp://localhost/myapp/mystream
```
3) To run main.py with limited resources:  
```
systemd-run --scope -p MemoryLimit=1000M -p CPUQuota=10% python3 main.py
```

4) In the main.py you can change the constant MODEL_PATH to put a higher accuracy model `model_lite[0-2].tflite`. Default model model_lite0.tflite

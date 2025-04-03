import os
import threading
import time
from datetime import datetime

from octorest import OctoRest

IMG_DIR = './images'
CAP_INTERVAL_MS = 10000
UPDATE_INTERVAL = 3  # Time in seconds to update the interface display
REMOTE_SERVER_IP = "campalme@128.153.28.135"
REMOTE_SERVER_DEST = '/home/campalme/layer_shifting_detection/capture'


shift_detected = False

def make_client(url, apikey):
     """Creates and returns an instance of the OctoRest client.

     Args:
         url - the url to the OctoPrint server
         apikey - the apikey from the OctoPrint server found in settings
     """
     try:
         client = OctoRest(url=url, apikey=apikey)
         return client
     except ConnectionError as ex:
         # Handle exception as you wish
         print(ex)


def capture():

    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR, exist_ok=True)

    while True:
        now = datetime.now()
        dtstr = now.strftime("%Y-%m-%d_%H_%M_%S")
        os.system('raspistill -o ' + os.path.join(IMG_DIR, dtstr + '.jpg') + ' -t ' + str(CAP_INTERVAL_MS) + ' --width 3280 --height 2464 > /dev/null 2>&1')
        # time.sleep(CAP_INTERVAL_MS//1000)

def upload():
    while True:
        images = os.listdir(IMG_DIR)
        if len(images) != 0:
            for image in images:
                os.system('scp ' + os.path.join(str(IMG_DIR), image) + ' ' + str(REMOTE_SERVER_IP) + ':' + str(REMOTE_SERVER_DEST))
                os.system('rm ' + os.path.join(str(IMG_DIR), image))




if __name__ == '__main__':

    # Start capture thread
    cap_thread = threading.Thread(target=capture)
    upload_thread = threading.Thread(target=upload)
    
    cap_thread.start()
    upload_thread.start()

    status = ''
    error_detected = False

    client = make_client('http://128.153.134.177', 'F03049DA4FAB4C7896F29AA1726842C9')

    print(client.job_info()['progress'])

    # exit()

    while True:
        # Update variables
        status = client.printer()['state']['text']
        tool_temp = client.printer()['temperature']['tool0']['actual']
        bed_temp = client.printer()['temperature']['bed']['actual']

        



        os.system('clear')
        print('Status: ' + str(status))
        print('Temperatures')
        print('\tHotend Temp: ' + str(tool_temp))
        print('\tBed Temp: ' + str(bed_temp))

        if status == 'Printing':
            print_progress = round(client.job_info()['progress']['completion'], 2)
            time_elapsed = round(client.job_info()['progress']['printTime'] / 100, 2)
            time_remaining = round(client.job_info()['progress']['printTimeLeft'] / 100, 2)

            print('\nPrint Progress: ' + str(print_progress))
            print('\tTime Elapsed: ' + str(time_elapsed) + ' minutes')
            print('\tEstimated Time Remaining: ' + str(time_remaining) + ' minutes\n')
            print('Error Detected: ' + str(error_detected))
        time.sleep(UPDATE_INTERVAL)
    



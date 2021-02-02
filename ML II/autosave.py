import schedule 
import datetime, time, subprocess

def autosave():
    subprocess.call(['/bin/bash', '-i', '-c', 'cnp'])
    print('Pushed', datetime.datetime.now())

if __name__ == '__main__':
    schedule.every(5).minutes.do(autosave)
    while True:
        schedule.run_pending() 
        time.sleep(1)
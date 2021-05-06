""" Helper logging script. """

from datetime import datetime as date

def write_to_file(text, log_file):
    with open(f"./{log_file}", "a+") as f:
        timestamp = date.now().strftime("%d/%m/%Y - %H:%M:%S")
        f.write(f"[{timestamp}] {text}\n")

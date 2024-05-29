import time

def PrintWithBorders(info, border_char='-', width=50):
    """
    This function is used for printing information with border
    """
    border = border_char * width
    print(border)
    content_width = width - 4
    words = info.split()
    line = ''
    
    for word in words:
        if len(line) + len(word) + 1 > content_width:
            print('| ' + line.ljust(content_width) + ' |')
            line = word
        else:
            line += (' ' + word if line else word)
            
    if line:
        print('| ' + line.ljust(content_width) + ' |')
    
    print(border)
    

def GetTime():
    current_time = time.localtime()
    year = current_time.tm_year
    month = current_time.tm_mon
    day = current_time.tm_mday
    hour = current_time.tm_hour
    minute = current_time.tm_min
    second = current_time.tm_sec
    formatted_time = f"{year}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}_{second:02d}"

    return formatted_time
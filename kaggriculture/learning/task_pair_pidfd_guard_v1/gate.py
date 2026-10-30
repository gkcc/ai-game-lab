"""Trusted pre-exec gate: EOF or timeout exits before the command can fork."""

import os
import select
import sys


def main():
    try:
        if len(sys.argv) < 4:
            os._exit(76)
        fd = int(sys.argv[1])
        seconds = float(sys.argv[2])
        command = sys.argv[3:]
        if fd < 3 or not 0 < seconds <= 86_400 or not command or not os.path.isabs(command[0]):
            os._exit(76)
        if not select.select([fd], [], [], seconds)[0]:
            os._exit(74)
        permit = os.read(fd, 1)
        os.close(fd)
        if permit != b'G':
            os._exit(73)
        os.execv(command[0], command)
    except BaseException:
        os._exit(76)


if __name__ == '__main__':
    main()

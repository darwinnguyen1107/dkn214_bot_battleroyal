DARWIN SURVIVOR V8 - FIXED PACKAGE

IMPORTANT: Copy mechfight.py from the original kit's python folder into this folder before building.
Source: D:\Purdue Courses\Hackathon\mechfight-kit\python\mechfight.py

This package fixes the navigation module filenames and Docker entrypoint.
It does NOT include mechfight.py, which was not among the uploaded files.

PowerShell commands:
cd "D:\Purdue Courses\Hackathon\Survival bot\dkn214_bot_battleroyal-main"
Copy-Item "D:\Purdue Courses\Hackathon\mechfight-kit\python\mechfight.py" .
# Replace the existing Dockerfile, Examplev8.py, and navigation files with the files from this package.
docker build --no-cache -t my-bot .
cd "D:\Purdue Courses\Hackathon\mechfight-kit"
python check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "${PWD}/arena:/arena:ro" my-bot

Optional: Build nav_cache.bin before Docker build, from bot folder:
python build_nav.py "D:\Purdue Courses\Hackathon\mechfight-kit\arena"
This can take time; do not run it if you are short on time.

The package was syntax-checked but not run through the official checker here.

# Realtime NN Classification with OpenVINO

A lightweight neural network classification software built in collaboration with Gary Baugh (Intel Application Engineer), using Intel's OpenVINO software. Allows for GPU- or CPU- load. AI models used include YOLO V3, MobileNet V2, and ResNet 50. Completed as a course requirement for TCD CSU33013 (Software Engineering Group Project), Spring Semester 2023.

Potential applications and use cases that build off of this software include advanced object detection/tracking, human distinction/facial recognition, hazard detection in automotive and industrial applications, etc. The real-time capabilities of the software, when combined with the options offered by the multiple models used and run on a sufficient system, allow for a high drgree of veresatility in potential future use cases.

## Authored by

- Andrii Yupyk
- Garrison Mullen
- James Fenlon
- Juliana Murphy
- Karolina Raczyńska
- Liam Junkermann
- Mykhailo Bitiutskyy
- Pierce Buckley
- Ryan Idowu
- Tadhg Brennan

<img src="https://github.com/mullen-zen/img-for-nn-readme/blob/main/photo.jpg?raw=true" alt="Group Photo" width="50%" height="50%">

## Setup

### LFS Init

``` {.sh}
git lfs install
git lfs pull
```

### Virtual Environment Setup/Activation

``` {.sh}
python -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Windows setup

``` {.cmd}
python -m venv .venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the sofware

``` {.cmd}
python /src/gui.py
```

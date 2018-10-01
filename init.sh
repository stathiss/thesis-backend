#!/bin/bash

# From current directory
cd /home/george/Desktop/thesis

# Start Virtual Environment
source venv/bin/activate

# Install pip
pip install pip==18.0

# Install requirements
pip install -r requirements.txt

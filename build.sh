#!/bin/bash

# Install Java
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk

# Install Python dependencies
pip install -r requirements.txt

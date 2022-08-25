#!/bin/bash

echo -e "\n downloading data...\n"
wget https://www.dropbox.com/s/f0b9uubjf3l1bdy/woodshole.zip
unzip woodshole.zip
rm woodshole.zip

echo -e "\n creating conda env... \n"
conda env create -f environment.yml
jupyter nbextension enable --py widgetsnbextension

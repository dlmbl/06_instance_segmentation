echo -e "\n creating mamba env... \n"
mamba env create -f environment.yml
jupyter nbextension enable --py widgetsnbextension
mamba activate 06_instance_segmentation

echo -e "\n downloading data...\n"
aws s3 cp "s3://dl-at-mbl-2023-data/woodshole_new.zip" "." --no-sign-request
unzip woodshole_new.zip
rm woodshole_new.zip
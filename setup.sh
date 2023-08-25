mamba activate base
echo -e "\n creating mamba env... \n"
mamba env create -f environment.yml
jupyter nbextension enable --py widgetsnbextension
mamba activate base

echo -e "\n downloading data...\n"
aws s3 cp "s3://dl-at-mbl-2023-data/woodshole_new.zip" "." --no-sign-request
unzip woodshole_new.zip
mkdir woodshole
mv woodshole_new/* woodshole
rm woodshole_new.zip
rm -r woodshole_new

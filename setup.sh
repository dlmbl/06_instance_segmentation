# Create environment name based on the exercise name
conda create -n 04-instance-segmentation python=3.11 -y
conda activate 04-instance-segmentation

# Install additional requirements
pip install uv
uv pip install -r requirements.txt

# Return to base environment
conda deactivate

# Download and extract data, etc.
echo -e "\n downloading data...\n"
aws s3 cp "s3://dl-at-mbl-2023-data/woodshole_new.zip" "." --no-sign-request
unzip woodshole_new.zip
mkdir woodshole
mv woodshole_new/* woodshole
rm woodshole_new.zip
rm -r woodshole_new

ROOT_FILE_URL := "https://nas.infra.es.nsu.ru/share.cgi?ssid=0BJWiEx&fid=0BJWiEx&filename=farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root&openfolder=forcedownload&ep="
DATASET_DIR := dataset
ROOT_FILENAME := farichsim_pi-pi-_45-360deg_1200.0k_ideal_2020-12-24_rndm.root
download_root_file:
	ls ${DATASET_DIR}/${ROOT_FILENAME} > /dev/null  ||  wget -O ${DATASET_DIR}/${ROOT_FILENAME} ${ROOT_FILE_URL}

setup_venv:
	(cat .venv/bin/activate > /dev/null || python3.8 -m venv .venv) \
	&& source .venv/bin/activate \
	&& pip install --upgrade pip \
	&& pip install -r requirements.txt

build: setup_venv download_root_file
	echo "Use venv: source .venv/bin/activate"


clear: 
	rm -rf dataset/graph_prew/*/processed
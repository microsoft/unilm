# Data Preparation

## InstructPix2Pix
```shell
bash scripts/download_data.sh path/to/clip-filtered-dataset
python convert_instructp2p.py --data-dir /path/to/clip-filtered-dataset/ --output-dir /path/to/output-dir/ --num-process 64
```

## OpenImage
```shell
wget https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv
python convert_openimage.py --data-dir /path/to/image_ids_and_rotation.csv --output-dir /path/to/output-dir/ --num-process 8 --cuda_device [0, 1, 2, 3, 4, 5, 6, 7]
```

if you want to preprocess the data in multiple nodes, you need to specify the `--num-machine` and `--machine-id` arguments. For example, if you want to preprocess the data in 8 nodes, you can run the following command in node 0:
```shell
python convert_openimage.py --data-dir /path/to/image_ids_and_rotation.csv --output-dir /path/to/output-dir/ --num-process 8 --cuda_device [0, 1, 2, 3, 4, 5, 6, 7] --num-machine 8 --machine-id 0
```
and run the following command in node 1:
```shell
python convert_openimage.py --data-dir /path/to/image_ids_and_rotation.csv --output-dir /path/to/output-dir/ --num-process 8 --cuda_device [0, 1, 2, 3, 4, 5, 6, 7] --num-machine 8 --machine-id 1
```
and so on.
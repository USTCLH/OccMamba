DATASET='kitti'
WORKERS=4

python -u tools/create_data.py ${DATASET} \
            --root-path data/${DATASET} \
            --out-dir data/${DATASET} \
            --workers ${WORKERS} \
            --extra-tag ${DATASET}
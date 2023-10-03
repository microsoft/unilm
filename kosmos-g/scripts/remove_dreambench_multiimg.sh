#!/bin/bash

DATASET_PATH=${1}

declare -a files_to_keep=(
"${DATASET_PATH}/pink_sunglasses/04.jpg"
"${DATASET_PATH}/can/01.jpg"
"${DATASET_PATH}/candle/02.jpg"
"${DATASET_PATH}/teapot/04.jpg"
"${DATASET_PATH}/backpack/02.jpg"
"${DATASET_PATH}/dog3/05.jpg"
"${DATASET_PATH}/cat2/02.jpg"
"${DATASET_PATH}/dog8/04.jpg"
"${DATASET_PATH}/grey_sloth_plushie/04.jpg"
"${DATASET_PATH}/backpack_dog/02.jpg"
"${DATASET_PATH}/robot_toy/00.jpg"
"${DATASET_PATH}/dog5/00.jpg"
"${DATASET_PATH}/duck_toy/01.jpg"
"${DATASET_PATH}/dog2/02.jpg"
"${DATASET_PATH}/dog/02.jpg"
"${DATASET_PATH}/colorful_sneaker/01.jpg"
"${DATASET_PATH}/red_cartoon/00.jpg"
"${DATASET_PATH}/clock/03.jpg"
"${DATASET_PATH}/shiny_sneaker/01.jpg"
"${DATASET_PATH}/dog6/02.jpg"
"${DATASET_PATH}/berry_bowl/02.jpg"
"${DATASET_PATH}/bear_plushie/03.jpg"
"${DATASET_PATH}/poop_emoji/00.jpg"
"${DATASET_PATH}/rc_car/03.jpg"
"${DATASET_PATH}/dog7/01.jpg"
"${DATASET_PATH}/cat/04.jpg"
"${DATASET_PATH}/fancy_boot/02.jpg"
"${DATASET_PATH}/wolf_plushie/04.jpg"
"${DATASET_PATH}/monster_toy/04.jpg"
"${DATASET_PATH}/vase/02.jpg"
)

find "${DATASET_PATH}" -type f | while read file; do
    keep=false
    for keep_file in "${files_to_keep[@]}"; do
        if [[ "${file}" == "${keep_file}" ]]; then
            keep=true
            break
        fi
    done

    if [[ "${keep}" == "false" ]]; then
        rm "${file}"
        echo "Deleted: ${file}"
    fi
done

echo "Cleanup completed!"

#!/bin/bash

function get_xml_total_mass() {
    urdf_file=$1
    grep -Pzo '(?s)mass\s*[^>]*?="\K[0-9.]+' "$urdf_file" | tr '\0' '\n' | awk '{sum += $1} END {printf "%.4f\n", sum}'
}

get_xml_total_mass "$1"

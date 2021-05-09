#!/bin/bash
new_path="saved_weights/$1/$2/log.txt"
echo $new_path
mv log.txt $new_path
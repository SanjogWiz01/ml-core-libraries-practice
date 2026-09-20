#!/bin/bash
cd "C:\Users\sanjo\OneDrive\Attachments\Desktop\Coding Shit\ml-core-libraries-practice" || exit 1

for i in $(seq 1 10); do
    echo "Checkpoint $i" > "checkpoint_$i.txt"
    git add -A
    git commit -m "Checkpoint $i: ML model development progress"
    git push origin models_train
    echo "Push $i complete"
done
echo "All 10 checkpoints pushed successfully!"
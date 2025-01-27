#!/bin/bash

STUDY_NAME="study_loss_calerror_2"

git pull
rm -rf logs/*
cd notebooks
rm -rf logs/*

mysql -u root -e "CREATE DATABASE IF NOT EXISTS $STUDY_NAME;"
optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/$STUDY_NAME" --directions ["minimize" "minimize"]


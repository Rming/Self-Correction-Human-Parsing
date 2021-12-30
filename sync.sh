#!/bin/bash
rsync -avz \
--exclude='pretrain_model' \
--exclude='.DS_Store' \
--exclude='.git/' \
--exclude='__pycache__/' \
./* gpu:tf/hp

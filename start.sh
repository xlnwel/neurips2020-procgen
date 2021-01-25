#!/bin/zsh

tag="Nov10-1954"

cd ~/SageMaker/proc
docker build -t interim .
cd ~/SageMaker/proc/docker
chmod +x entrypoint.sh
docker build -t "758410179420.dkr.ecr.us-west-2.amazonaws.com/procgen:${tag}" .
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 758410179420.dkr.ecr.us-west-2.amazonaws.com
docker push "758410179420.dkr.ecr.us-west-2.amazonaws.com/procgen:${tag}"
cd ~/SageMaker/proc
pip install sagemaker==2.6.0
sed -e "s/TAG/${tag}/g" aws.py > run.py

python run.py
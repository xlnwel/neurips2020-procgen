#!/bin/zsh

# Build your normal docker image
docker build --tag experiment-interim .

# Make some changes to make it work with sagemaker
mkdir -p /tmp/docker

cat <<EOF > /tmp/docker/Dockerfile
FROM experiment-interim
USER root
RUN echo "#!/bin/bash\n" > /entrypoint.sh \
 && echo "chmod +x /opt/ml/input/data/inputs/run.sh" >> /entrypoint.sh \
 && echo "/opt/ml/input/data/inputs/run.sh" >> /entrypoint.sh \
 && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
EOF

cd /tmp/docker
docker build -t 758410179420.dkr.ecr.us-west-2.amazonaws.com/procgen:evaluation .

# Login to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 758410179420.dkr.ecr.us-west-2.amazonaws.com

# Push the image to ECR
docker push 758410179420.dkr.ecr.us-west-2.amazonaws.com/procgen:evaluation

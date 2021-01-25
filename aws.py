import sagemaker
from sagemaker.estimator import Estimator
import uuid

s3_bucket = "procgen"
run_id = uuid.uuid4().hex[:8]

def create_sm_estimator(env_name, mode="train") -> Estimator:
    return Estimator(
        image_uri="758410179420.dkr.ecr.us-west-2.amazonaws.com/procgen:evaluation",
        role="arn:aws:iam::758410179420:role/sm-soln-rl-procgen-neuips-NotebookInstanceExecutio-1ADT1VW48LFM7",
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        use_spot_instances=True,
        max_wait=10000,
        max_run=7650,
        volume_size=30,
        base_job_name=f"{run_id}",
        output_path=f"s3://{s3_bucket}/outputs/{run_id}/{mode}/{env_name}",
        checkpoint_local_path="/ray-outputs",
        checkpoint_s3_uri=f"s3://{s3_bucket}/checkpoints/{run_id}/{mode}/{env_name}",
    )

def train(env_name):
    estimator = create_sm_estimator(env_name, "train")
    job_name = estimator.base_job_name + f"-train-{env_name}"
    estimator.set_hyperparameters(mode="train")
    estimator.fit(job_name=job_name)

def rollout(env_name):
    estimator = create_sm_estimator(env_name, "rollout")
    job_name = estimator.base_job_name + f"-rollout-{env_name}"
    estimator.set_hyperparameters(mode="rollout")
    estimator.fit(
        inputs={
                "chkpts": f"s3://{s3_bucket}/checkpoints/{run_id}/train/{env_name}/",
            },
        job_name=job_name,
    )

for env_name in ["chaser"]:
    train(env_name)

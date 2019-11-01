import sagemaker
import argparse
from sagemaker.tensorflow import TensorFlow
parser = argparse.ArgumentParser()
parser.add_argument("--node-count", type=int, help="number of nodes to train", default=4)
parser.add_argument("--python", help="python version", default="py3")

args = parser.parse_args()

sagemaker_session = sagemaker.Session()

tf_estimator = TensorFlow(
    sagemaker_session=sagemaker_session,
    script_mode=True,
    entry_point="singletrain.sh",
    source_dir="../benchmarks/tr-cpu/tf",
    role="SageMakerRole",
    train_instance_count=args.node_count,
    train_instance_type="ml.c5.18xlarge",
    image_name="841569659894.dkr.ecr.us-east-1.amazonaws.com/beta-tensorflow-training:1.15.0-" + args.python + "-cpu-with-horovod-build",
    #image_name="841569659894.dkr.ecr.us-east-1.amazonaws.com/beta-tensorflow-training:1.13-py3-cpu-with-horovod-build-2019-05-25-00-41-18",
    #image_name="520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tensorflow-scriptmode:1.13-cpu-py3",
    py_version=args.python,
    framework_version="1.15.0",
    distributions={
        "mpi": {
            "enabled": True,
            "processes_per_host": 1,
            "custom_mpi_options": "-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 -x TF_CPP_MIN_LOG_LEVEL=0",
        }
    },
    output_path="s3://asimov-bai-results-sagemaker"
    # subnets=["subnet-07735e63c73eddfc0", "subnet-0c027b8eafad8d482"],
    # subnets=["subnet-07735e63c73eddfc0"],
    # security_group_ids=["sg-0a2531f240064758a", "sg-03a2f31c5c8cd5a39"],
)

data = {
    # Just to make sm happy
    "s1": "s3://mxnet-asimov-data-sagemaker/small"
}

tf_estimator.fit(data, logs=True, wait=True)

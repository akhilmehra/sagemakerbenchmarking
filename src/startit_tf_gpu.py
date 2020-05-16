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
    source_dir="../benchmarks/tr-gpu/tf",
    role="SageMakerRole",
    train_instance_count=args.node_count,
    train_instance_type="ml.p3.16xlarge",
    image_name="841569659894.dkr.ecr.us-east-1.amazonaws.com/beta-tensorflow-training:2.1.0-" + args.python + "-gpu-with-horovod-build",
    py_version=args.python,
    framework_version="2.1.0",
      distributions={
          "mpi": {
              "enabled": True,
              "processes_per_host": 8,
              "custom_mpi_options": "-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 -x TF_CPP_MIN_LOG_LEVEL=0",
          }
      },
    output_path="s3://asimov-bai-results-sagemaker",
    train_volume_size=200
    #subnets=["subnet-07735e63c73eddfc0", "subnet-0c027b8eafad8d482"],
    # subnets=["subnet-07735e63c73eddfc0"],
    # security_group_ids=["sg-0a2531f240064758a", "sg-03a2f31c5c8cd5a39"]
)

data = {
    "train": "s3://mxnet-asimov-data-sagemaker/imagenet/raw/train-480px",
    "validate": "s3://mxnet-asimov-data-sagemaker/imagenet/raw/validation-480px",
}

tf_estimator.fit(data, logs=True, wait=True)

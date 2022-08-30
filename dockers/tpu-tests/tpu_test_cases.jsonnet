local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'pl',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 6000, # 100 minutes, in seconds.

  image: 'pytorchlightning/pytorch_lightning',
  imageTag: 'base-xla-py{PYTHON_VERSION}-torch{PYTORCH_VERSION}',

  tpuSettings+: {
    softwareVersion: 'pytorch-{PYTORCH_VERSION}',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      source ~/.bashrc
      conda activate lightning
      mkdir -p /home/runner/work/lightning-flash && cd /home/runner/work/lightning-flash
      git clone https://github.com/Lightning-AI/lightning-flash.git
      cd lightning-flash
      echo $PWD
      git ls-remote --refs origin
      git fetch origin "refs/pull/{PR_NUMBER}/head:pr/{PR_NUMBER}" && git checkout "pr/{PR_NUMBER}"
      git checkout {SHA}
      export FREEZE_REQUIREMENTS=1
      pip install -e .[test]
      echo $KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
      export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"
      export FLASH_RUN_TPU_TESTS=1
      cd tests
      coverage run --source=lightning_flash -m pytest -vv --durations=0 ./
      echo "\n||| Running TPU Tests |||\n"
      bash tpu_tests.sh
      test_exit_code=$?
      echo "\n||| END PYTEST LOGS |||\n"
      coverage xml
      cat coverage.xml | tr -d '\t'
      test $test_exit_code -eq 0
    |||
  ),
};

tputests.oneshotJob

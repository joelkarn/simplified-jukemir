source env.sh

pushd ..
set -e
HOST_CACHE=$(python -c "from jukemir import CACHE_DIR; print(CACHE_DIR)")
echo $HOST_CACHE
popd

docker run \
  -it \
  --rm \
  -d \
  ${DOCKER_CPU_ARG} \
  ${DOCKER_GPU_ARG} \
  --name ${DOCKER_NAME} \
  -v $(pwd)/../jukemir:/jukemir/jukemir \
  -v $HOST_CACHE:/jukemir/cache \
  -v $(pwd)/../tests:/jukemir/tests \
  -v $(pwd)/../notebooks:/jukemir/notebooks \
  -v $(pwd)/../scripts:/jukemir/scripts \
  -p 8888:8888 \
  ${DOCKER_NAMESPACE}/${DOCKER_TAG} \
  bash

ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
FROM ${ISAAC_LAB_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MINI_PI0_IN_ISAACLAB_CONTAINER=1

WORKDIR /workspace/mini-pi0

COPY docker/isaaclab/entrypoint.sh /usr/local/bin/mini-pi0-isaaclab-entrypoint
RUN chmod +x /usr/local/bin/mini-pi0-isaaclab-entrypoint

ENTRYPOINT ["/usr/local/bin/mini-pi0-isaaclab-entrypoint"]
CMD ["bash"]

FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.8

RUN export HTTP_PROXY="http://192.168.49.1:8282" \
    && export http_proxy="http://192.168.49.1:8282" \
    https_proxy="http://192.168.49.1:8282" \
    && apt-get update \
    && unset HTTP_PROXY http_proxy https_proxy

RUN export HTTP_PROXY="http://192.168.49.1:8282" \
    && export http_proxy="http://192.168.49.1:8282" \
    https_proxy="http://192.168.49.1:8282" \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && echo "Acquire { HTTP::proxy \"http://192.168.49.1:8282\"; HTTPS::proxy \"http://192.168.49.1:8282\"; }" > /etc/apt/apt.conf \
    # && apt install --no-install-recommends -y <packages-here> \
    && unset HTTP_PROXY http_proxy https_proxy

COPY .devcontainer/requirements.txt ./

RUN export HTTP_PROXY="http://192.168.49.1:8282" \
    && export http_proxy="http://192.168.49.1:8282" \
    https_proxy="http://192.168.49.1:8282" \
    && pip3 --no-cache-dir install "ansible-lint[core,yamllint]" \
    && python3 -m pip install cython \
    && pip3 install --proxy=http://192.168.49.1:8282 --no-cache-dir -r requirements.txt

RUN pip3 install --proxy=http://192.168.49.1:8282 --no-cache-dir tensorflow==2.2.0
RUN pip3 install --proxy=http://192.168.49.1:8282 --no-cache-dir ipykernel==5.5.5

# TODO - install kite
# TODO - install git y todas sus extensiones


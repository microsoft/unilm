FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update 
RUN apt-get install -y python3 pip git sudo wget vim screen htop jq net-tools infiniband-diags perftest 

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /root/requirements.txt

RUN pip install -r /root/requirements.txt

WORKDIR ~

CMD ["/bin/bash"]
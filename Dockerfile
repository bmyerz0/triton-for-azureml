FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-py38-cuda11.7-gpu:4

RUN pip install matplotlib
RUN pip install triton

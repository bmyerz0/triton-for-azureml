# Triton on AzureML Studio

This project will help you get started using the [Triton programming language](https://triton-lang.org/main/index.html) on GPU resources in AzureML.

## Pre-requisites
- access to an AzureML Studio workspace, subscription
- compute cluster of Nvidia GPU resources

## Quickstart 
Use these instructions if you want to get something working as quickly as possible. There are comments in the code but not much is explained here, so if you want more understanding of AzureML, you might look at the tutorials in the Resources section below.

1. Go to [AzureML Studio](https://ml.azure.com/), go to your workspace, go to Notebooks
2. Create a folder called `triton-examples/`
3. Add the files/folders from this repository
5. Open triton_gpu_submit.ipynb
6. Fill in your subscription information
```
    subscription_id="YOUR-SUBSCRIPTION-ID",
    resource_group_name="YOUR-RESOURCE-GROUP-NAME",
    workspace_name="YOUR-WORKSPACE-NAME",
```
8. Create a Compute using the Set your kernel instructions from this tutorial. You just need to create an low-end CPU type that can run your notebook code. The actual Pytorch/Triton code will get run on your GPU cluster.
9. Do the instructions at Create a Triton Environment below
10. Where the script says `environment`, replace the value with the name and version of the Environment you created

```
environment="acpt-triton-2@latest", # what environment the compute cluster node will be running
```
9. In triton_gpu_submit.ipynb, run all the cells (runs on your cheap CPU compute). The last one actually submits your Triton/Pytorch job to the GPU cluster
10. Open the link that gets printed to see the status of your job
11. In that dashboard, when the job is finished you should the output of triton_gpu.py in Outputs + logs
12. To test a different Triton program, add it to the triton-examples/src folder and change this line of the submit script to call your python file.

```
    command="python activation.py",
```


## Create a Triton Environment

We do not yet have a hosted Environment with Triton installed. Here is how you can quickly create your own.

You'll need an Environment with Pytorch and Triton installed. Here is one way to do it with Azure Container for Pytorch (ACPT), but you can pick the base environment you prefer.
1. Follow these instructions: [How to create Azure Container for PyTorch Custom Curated environment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment?view=azureml-api-2)
2. Use the option of a Dockerfile, use the contents shown here TODO
3. Note the name you gave to the environment; you'll need to put it in your GPU submit script

## General AzureML Studio resources

- [Azure Machine Learning documentation | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2)
- see the Tutorial cards on the Home page of AzureML Studio
- repo of examples/tutorials: [azureml-examples/tutorials at main · Azure/azureml-examples (github.com)](https://github.com/Azure/azureml-examples/tree/main/tutorials)
- [How to create a compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster)	
- MLFlow
  - [Quickstart: Install MLflow, instrument code & view results in minutes — MLflow 2.3.1 documentation](https://mlflow.org/docs/latest/quickstart.html)
  - MLFlow on AzureML: [MLflow and Azure Machine Learning - Azure Machine Learning | Microsoft Learn](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)

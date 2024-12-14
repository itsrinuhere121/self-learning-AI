Request ID: 815586393980789
How to download the model
Visit the Llama repository in GitHub where instructions can be found in the Llama README.
1
Install the Llama CLI
In your preferred environment run the command below:
Command
pip install llama-stack
Use -U option to update llama-stack if a previous version is already installed:
Command
pip install llama-stack -U
2
Find models list
See latest available models by running the following command and determine the model ID you wish to download:
Command
llama model list
If you want older versions of models, run the command below to show all the available Llama models:
Command
llama model list --show-all
3
Select a model
Select a desired model by running:
Command
llama model download --source meta --model-id  MODEL_ID
4
Specify custom URL
Llama 3.3: 70B
When the script asks for your unique custom URL, please paste the URL below
URL
https://llama3-3.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiYzdxYzhkYnhwYmc5bGdqOHNkbjQ3ZTVtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTMubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNDMzNTczNn19fV19&Signature=h3vgv5FZnfe5UJ2Pj6dLQ4uOAU-dtE%7Er6LLXmEiBHGIpUcOlYCcfUVl%7E%7EOg28q1aoZSmMxqck%7EQRYEwyivKteXCGzH5kqmnbSsIxv5F9e8RIPaFg-TQS81ma169likSslDKOT9X9QG4sfkfGsU4oKSHZNqST7u2Js0EIW4wFtb8vzO1%7E15r08tsKiYGfRlLLRLUb7qSWPUZTVSa0GzRRfNTH594xDjhYSe5OMFd415RWV2nuuPYR7ceIMkxDkqa%7E0LyBfk3i1aNhDVrYxNautaIsrxq3NZqvoqy-mnkG-ttJBhAt--oNylMI1NspXlG9QmQnoAAuKSWs8te8IoRBzA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=623406183589545
Please save copies of the unique custom URLs provided above, they will remain valid for 48 hours to download each model up to 5 times, and requests can be submitted multiple times. An email with the download instructions will also be sent to the email address you used to request the models.

Available models
The Llama 3.3 70B includes instruct weights only. Instruct weights have been fine-tuned and aligned to follow instructions. They can be used as-is in chat applications or further fine-tuned and aligned for specific use cases.


Available models for download include:

Fine-tuned:
Llama-3.3-70B-Instruct
gcloud config set project <project_real_name>

gcloud builds submit --tag gcr.io/ml-22-crypto-fraud-detection/backendv1  --project=ml-22-crypto-fraud-detection
gcloud run deploy --image gcr.io/ml-22-crypto-fraud-detection/backendv1 --platform managed  --project=ml-22-crypto-fraud-detection --allow-unauthenticated

gcloud builds submit --tag gcr.io/crypto-fraud-detection/backendv1  --project=crypto-fraud-detection
gcloud run deploy --image gcr.io/crypto-fraud-detection/backendv1 --platform managed  --project=crypto-fraud-detection --allow-unauthenticated

checkout backend_deployment_branch
> python run main.py

requirements for firebase
https://stackoverflow.com/questions/58354509/modulenotfounderror-no-module-named-python-jwt-raspberry-pi
https://www.youtube.com/watch?v=K4jnYNU9RfA

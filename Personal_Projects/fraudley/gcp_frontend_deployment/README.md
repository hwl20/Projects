gcloud builds submit --tag gcr.io/ml-22-crypto-fraud-detection/frontendv1  --project=ml-22-crypto-fraud-detection
gcloud run deploy --image gcr.io/ml-22-crypto-fraud-detection/frontendv1 --platform managed  --project=ml-22-crypto-fraud-detection --allow-unauthenticated

gcloud builds submit --tag gcr.io/crypto-fraud-detection/frontendv1  --project=crypto-fraud-detection
gcloud run deploy --image gcr.io/crypto-fraud-detection/frontendv1 --platform managed  --project=crypto-fraud-detection --allow-unauthenticated

checkout frontend_deployment_branch
> streamlit run main.py
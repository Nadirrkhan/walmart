import boto3
import botocore

session = boto3.Session()
creds = session.get_credentials()
print("Access Key:", creds.access_key)
print("Secret Key:", creds.secret_key)

# Check region
print("Region:", session.region_name)

# Try listing the bucket
try:
    s3 = session.client("s3")
    response = s3.list_objects_v2(Bucket="mlops-testing-1", Prefix="mlops-1/")
    print("Objects in S3 path:")
    for obj in response.get("Contents", []):
        print(" -", obj["Key"])
except botocore.exceptions.ClientError as e:
    print("Error:", e)

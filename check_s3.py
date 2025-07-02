import boto3

# Create the S3 client
s3 = boto3.client("s3")

# List objects in the given bucket and prefix
response = s3.list_objects_v2(Bucket="mlops-testing-1", Prefix="mlops-1/")

# Print the object keys
print("Objects in S3 path:")
for obj in response.get("Contents", []):
    print(" -", obj["Key"])

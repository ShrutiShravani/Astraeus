import boto3
import os
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_directory, bucket_name, s3_prefix="pdfs/"):
    s3 = boto3.client('s3')
    
    for filename in os.listdir(local_directory):
        if filename.endswith(".pdf"):
            local_path = os.path.join(local_directory, filename)
            s3_path = os.path.join(s3_prefix, filename)
            
            try:
                print(f"Uploading {filename} to S3...")
                s3.upload_file(local_path, bucket_name, s3_path)
            except FileNotFoundError:
                print("The file was not found")
            except NoCredentialsError:
                print("Credentials not available")

if __name__=="__main__":
    upload_to_s3(local_directory="data\raw",bucket_name="astraeu",s3_prefix="pdfs/")
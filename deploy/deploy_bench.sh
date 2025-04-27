#!/bin/bash
set -euo pipefail

INSTANCE_ID=$( \
  aws ec2 run-instances \
    --image-id "ami-0e35ddab05955cf57" \
    --instance-type "c5a.4xlarge" \
    --key-name "fss_bench" \
    --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-00a5570388877549e","VolumeSize":8,"VolumeType":"gp3","Throughput":125}}' \
    --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-0cabb34934ef4b73c"]}' \
    --tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"fss_bench"}]}' \
    --metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
    --private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":true,"EnableResourceNameDnsAAAARecord":false}' \
    --count "1" \
    --user-data file://deploy/userdata.txt \
    --output json | \
  jq -r '.Instances[0].InstanceId' \
)
echo "Instance created"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
INSTANCE_ADDR=$( \
  aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
)

set +e
ssh -T -i ~/.ssh/aws_ec2/fss_bench -o StrictHostKeychecking=no ubuntu@"$INSTANCE_ADDR" < deploy/wait_file_exists.sh
if [ $? -ne 0 ]; then
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
  echo "Instance terminated"
  exit 1
fi
set -e
echo "Bench finished"

rsync -e "ssh -i ~/.ssh/aws_ec2/fss_bench -o StrictHostKeychecking=no" ubuntu@"$INSTANCE_ADDR":/tmp/fss_bench.log .

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
echo "Instance terminated"

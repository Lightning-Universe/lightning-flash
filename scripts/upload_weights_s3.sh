# This script is an utils to simplify updating the weights on s3
# >>> python flash_examples/finetuning/...
# >>> sh scripts/upload_weights_s3.sh
# The weights will be deleted after upload.
files=$(ls . | grep .pt)
for f in $files
do
	echo "uploading $f"
	aws s3 cp $f s3://flash-weights/$f --acl public-read
	rm $f
done

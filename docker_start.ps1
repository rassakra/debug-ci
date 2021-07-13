echo "Starting NLP server on port 5000"
docker run --rm -p 5000:5000 -it $(docker build -q .)

# use -d to run in detatched mode
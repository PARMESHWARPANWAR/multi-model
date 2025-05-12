python -m venv multi_model_env

multi_model_env\Scripts\activate   (Windows)

source multi_model_env/bin/activate (Linux)

pip install -r requirements.txt
pip install pandas
pip install soundfile     (Windows)


uvicorn app.main:app --reload


Local mongoDB initialization

python3.10.9


# Build the images
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop the services
docker-compose down

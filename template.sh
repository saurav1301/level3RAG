# create the folders 
mkdir -p app
mkdir -p data
mkdir -p docker


#  create the files 

touch app/__init__.py
touch app/agent_builder.py
touch app/tools.py
touch app/rag_server.py
touch app/settings.py


touch docker/Dockerfile
touch docker/start.sh



touch requirements.txt
touch .env


echo "project file created successfully."
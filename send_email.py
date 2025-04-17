import os
from fastapi import BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv('.env')

class Envs:
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_FROM = os.getenv('MAIL_FROM')
    MAIL_PORT = int(os.getenv('MAIL_PORT'))
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_FROM_NAME = os.getenv('MAIL_FROM_NAME')

# Configure the FastAPI-Mail connection
# conf = ConnectionConfig(
#     MAIL_USERNAME=Envs.MAIL_USERNAME,
#     MAIL_PASSWORD=Envs.MAIL_PASSWORD,
#     MAIL_FROM=Envs.MAIL_FROM,
#     MAIL_PORT=Envs.MAIL_PORT,
#     MAIL_SERVER=Envs.MAIL_SERVER,
#     MAIL_FROM_NAME=Envs.MAIL_FROM_NAME,
#     MAIL_TLS=True,
#     MAIL_SSL=False,
#     USE_CREDENTIALS=True,
#     TEMPLATE_FOLDER='./templates/email'
# )
conf = ConnectionConfig(
    MAIL_USERNAME=Envs.MAIL_USERNAME,
    MAIL_PASSWORD=Envs.MAIL_PASSWORD,
    MAIL_FROM=Envs.MAIL_FROM,
    MAIL_PORT=Envs.MAIL_PORT,
    MAIL_SERVER=Envs.MAIL_SERVER,
    MAIL_FROM_NAME=Envs.MAIL_FROM_NAME,
    USE_CREDENTIALS=True,
    TEMPLATE_FOLDER='./templates/email',
    MAIL_STARTTLS=True,   # Replaces MAIL_TLS
    MAIL_SSL_TLS=False    # Replaces MAIL_SSL
)

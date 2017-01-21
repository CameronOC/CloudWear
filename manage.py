# This file starts the WSGI web application.
# - Heroku starts gunicorn, which loads Procfile, which starts manage.py
# - Developers can run it from the command line: python runserver.py

from app import create_app, manager
from flask_script import Server

app = create_app()
server = Server(host="127.0.0.1", port=80)
manager.add_command("runserver", Server())

# Start a development web server if executed from the command line
if __name__ == "__main__":
    # Manage the command line parameters such as:
    # - python manage.py runserver
    # - python manage.py db


    manager.run(host= '0.0.0.0')

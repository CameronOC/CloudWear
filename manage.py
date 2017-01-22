# manage.py


import os
import unittest
import coverage

from flask import send_from_directory

from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from source import app, db
from source.models import User


app.config.from_object(os.environ['APP_SETTINGS'])

migrate = Migrate(app, db)
manager = Manager(app)

# migrations
manager.add_command('db', MigrateCommand)

@app.route('/source/templates/assests/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/source/templates/assests/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/source/templates/assests/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory('fonts', path)

@app.route('/source/templates/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

@app.route('/source/templates/<path:path>')
def send_templates(path):
    return send_from_directory('templates', path)

@manager.command
def test():
    """Runs the unit tests without coverage."""
    tests = unittest.TestLoader().discover('tests')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    else:
        return 1


@manager.command
def cov():
    """Runs the unit tests with coverage."""
    cov = coverage.coverage(branch=True, include='project/*')
    cov.start()
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)
    cov.stop()
    cov.save()
    print('Coverage Summary:')
    cov.report()
    basedir = os.path.abspath(os.path.dirname(__file__))
    covdir = os.path.join(basedir, 'tmp/coverage')
    cov.html_report(directory=covdir)
    print('HTML version: file://%s/index.html' % covdir)
    cov.erase()


@manager.command
def create_db():
    """Creates the db tables."""
    db.create_all()


@manager.command
def drop_db():
    """Drops the db tables."""
    db.drop_all()


@manager.command
def create_admin():
    """Creates the admin user."""
    db.session.add(User("ad@min.com", "admin"))
    db.session.commit()


if __name__ == '__main__':
    manager.run()

# project/main/views.py


#################
#### imports ####
#################

from flask import render_template, Blueprint
from flask_login import login_required


################
#### config ####
################

main_blueprint = Blueprint('main', __name__,)


################
#### routes ####
################

@main_blueprint.route('/')
@login_required
def home():
    return render_template('main/index.html')

@main_blueprint.route('/upload/<key>', methods=['POST'])
@login_required
def upload():

    return